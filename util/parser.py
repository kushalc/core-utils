import logging
import os
import sys
import traceback

import ahocorasick as aho
import html2text
import numpy as np
import pandas as pd
import regex as re
import spacy
from sklearn.preprocessing import FunctionTransformer
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token

from util.caching import _cache_path
from util.performance import instrument_latency

# NOTE: ALL spaCy instances share the same vocab! This dramatically increases parse speed since
# spaCy internally stores strings as ints and not having to serialize/de-serialize the same strings
# repeatedly is a huge win for it.
VOCAB = True
_SPACY_CACHE = {}
def get_cached_spacy(**kwargs):
    global _SPACY_CACHE
    hashed = _cache_path(sys.modules[__name__], __get_uncached_spacy, tuple(), kwargs)
    if hashed in _SPACY_CACHE:
        return _SPACY_CACHE[hashed]

    global VOCAB
    # FIXME: Saving vocab independently of Language causes problems. In particular,
    # pos_ tags don't seem to be loaded properly.
    #
    # vocab_path = _cache_path(sys.modules[__name__], __get_uncached_spacy, tuple("vocab",), {})
    # if VOCAB is True and os.path.exists(vocab_path):
    #     logging.info("Loading Vocab from %s", vocab_path)
    #     _SPACY_CACHE[vocab_path] = VOCAB = Vocab().from_disk(vocab_path)
    _SPACY_CACHE[hashed] = __get_uncached_spacy(hashed, vocab=VOCAB, **kwargs)
    if VOCAB is True:
        VOCAB = _SPACY_CACHE[hashed].vocab
    return _SPACY_CACHE[hashed]

@instrument_latency
def __get_uncached_spacy(path, vocab=None, **kwargs):
    trace = " | ".join(x.replace("\t", " ").replace("\n", "\t") for x in traceback.format_stack())

    nlp = None
    if os.path.exists(path):
        try:
            logging.info("Loading from %s: %s; %s", path, kwargs, trace)
            nlp = Language(VOCAB).from_disk(path, **kwargs)
        except:
            logging.error("Couldn't load language from disk: %s", path, exc_info=True)

    if not nlp:
        nlp = spacy.load("en", vocab=vocab, **kwargs)

    return nlp

# NOTE: This is modified by ResumeParser::Timon. Don't use unless you expect it to be modified.
def _resume_spacy(**kwargs):
    return get_cached_spacy(disable=("ner",), **kwargs)

def _posting_spacy(**kwargs):
    return get_cached_spacy(disable=("parser", "ner",), **kwargs)

def _basic_spacy(**kwargs):
    return get_cached_spacy(disable=("parser", "ner", "tagger"), **kwargs)

class SpacyTransformer(FunctionTransformer):
    def __init__(self, func=_resume_spacy, field="text", kw_args={}):
        super(SpacyTransformer, self).__init__(func=func, validate=False, kw_args={})
        self.field = field

    def _generate(self, dataset):
        def _formed(x, classes=(Token, Doc, Span)):
            return isinstance(x, classes)
        if all(_formed(x, classes=(Doc, Span)) for x in dataset):
            return dataset

        def _transform_raw(x, was_df=False):
            if not was_df and isinstance(x, (pd.Series, dict)):
                x = x[self.field]
            return x

        def _transform_df(x):
            return _transform_raw(x, True)

        if isinstance(dataset, pd.DataFrame):
            values = dataset[self.field].apply(_transform_df)
        else:
            values = [_transform_raw(x) for x in dataset]

        return self.method(**self.kwargs).pipe([y.text_with_ws if _formed(y) else y
                                                for y in values])

    @instrument_latency
    def transform(self, dataset):
        return [x for x in self._generate(dataset)]

@instrument_latency
def build_phrase_matcher(spacy, iterable, label, length_max=9):
    matcher = PhraseMatcher(spacy.vocab)
     # FIXME: Temporary hack to make sure we don't add super-long things and cause below to break.
    _length = lambda x: len(x) if isinstance(x, (Doc, Span)) else len(x.split())
    iterable = [x for x in iterable if len(x) < length_max]
    iterable = OrderedSet(list(iterable) +
                          [x.title() for x in iterable] +
                          [x.lower() for x in iterable] +
                          [x.upper() for x in iterable])
     # NOTE: Apparently you can use the dereference operator with generators!
    # https://gist.github.com/sadovnychyi/90aa96a4dbaed71a466e82cc8ebe0a35#gistcomment-2208844
    matcher.add(label, None, *spacy.pipe(iterable))
    return matcher

_HTML_PARSER = None
def sanitize_html(raw, strict=True):
    global _HTML_PARSER
    if _HTML_PARSER is None:
        _HTML_PARSER = html2text.HTML2Text()
        _HTML_PARSER.ignore_emphasis = True
        _HTML_PARSER.ignore_links = True

    text = np.nan
    try:
        if pd.notnull(raw):
            if not strict:
                text = raw
            text = _HTML_PARSER.handle(raw).strip()
    except:
        logging.warn("Coudn't sanitize HTML: %.25s", raw, exc_info=True)
    return text

@instrument_latency
def build_aho_matcher(iterable, label):
    A = aho.Automaton()
    for ix, word in enumerate(iterable):
        A.add_word(word, (word, label))
    A.make_automaton()
    return A

def build_direct_matcher(iterable, flags=re.U, generous=False,
                         regexp=r"\b(%s)\b", length_min=2):
    if generous:
        matcher = re.compile(r"\W+")
        iterable = [re.sub(r"^\W+|\W+$", "", re.sub(r"\W+", r"\W+", wd)) for wd in iterable]

        # NOTE: Some non-English names get completely stripped in previous
        # regexp. Avoid matching entirely whitespace strings.
        iterable = [wd for wd in iterable if len(wd) > length_min]
    iterable = sorted(iterable, key=len, reverse=True) # NOTE: Want to be greedy about picking longest
                                                       # possible match.

    compiled = re.compile(regexp % "|".join(iterable), flags)
    return compiled

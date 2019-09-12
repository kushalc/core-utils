#! /usr/bin/env python

import email
import logging
import os
import re
import ssl
from collections import OrderedDict

import numpy as np
import pandas as pd

from imapclient import IMAPClient
from util.caching import cache_today
from util.performance import instrument_latency
from util.shared import first, parse_args


def capture_emails(query):
    imap = _setup_imap(os.environ["GMAIL_USERNAME"], os.environ["GMAIL_PASSWORD"])
    imap.select_folder("[Gmail]/All Mail")

    # @cache_today
    def __capture_emails(query):
        headers = imap.gmail_search(query)
        results = _fetch_emails(imap, headers)
        return results

    df = pd.DataFrame(__capture_emails(query))
    import pdb; pdb.set_trace()

    df = df.applymap(lambda x: x.decode("utf-8"))
    df.sort_values("received_on", ascending=False, inplace=True)

    import pdb; pdb.set_trace()

    def _fix_body(text):
        return re.sub(r"\s+", " ", re.sub("(\r|\n)", " ", text))
    df["body"] = df["body"].apply(_fix_body)

    return df

def _setup_imap(email_address, password):
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    imap = IMAPClient("imap.gmail.com", 993, ssl=True, ssl_context=ssl_context)
    imap.login(email_address, password)
    return imap

@instrument_latency
def _fetch_emails(imap, messages):
    def _email(address):
        return "%(mailbox)s@%(host)s" % address._asdict()

    def _name(address):
        return "%(name)s" % address._asdict()

    envelopes = imap.fetch(messages, ["ENVELOPE", "RFC822"])
    logging.info("Processing {} envelopes".format(len(envelopes)))

    def __parse_message(envelope, msg):
        base = {
            "received_on": envelope.date,
            "subject": envelope.subject,
        }

        body = ""
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += str(part.get_payload(decode=True))
        base["body"] = body

        if not envelope.sender:
            pass
        elif isinstance(envelope.sender, tuple):
            base["company_email"] = first(_email(addr) for addr in envelope.sender)
        else:
            base["company_email"] = _email(envelope.sender)

        return base

    results = []
    for id, envelope in envelopes.items():
        try:
            message = email.message_from_string(str(envelope.get(b"RFC822")))

            # TODO: Can we get the envelope items from the msg above?
            envelope = envelope[b"ENVELOPE"]
            base = __parse_message(envelope, message)

            recipients = sum(filter(None, [envelope.to, envelope.cc, envelope.bcc]), ())
            if not recipients:
                logging.warn("Skipped no-recipient message: %s", envelope)
                continue

            results += [dict(user_email=_email(addr), user_name=_name(addr), **base)
                             for addr in recipients]

        except AttributeError:
            logging.warn("Skipped erroneous message: %s", envelope, exc_info=True)
            import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    return results

if __name__ == "__main__":
    args = parse_args("Easily download emails from Gmail that meet criteria", [
         dict(name_or_flags="query", help="gmail-compatible query"),
    ])
    df = capture_emails(args.query)

    # columns = ["subject", "label", "user_name", "user_email", "company_email", "received_on", "body"]
    # output_path = os.path.join(args.output, "emails.%s.tsv" % os.environ["GMAIL_USERNAME"])
    # df.to_csv(output_path, columns=columns)
    import pdb; pdb.set_trace()

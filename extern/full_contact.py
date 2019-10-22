#! /usr/bin/env python

import json
import logging
import os
import urllib.error
import urllib.request

import pandas as pd

from util.caching import cache_today
from util.parallelization import parallel_apply
from util.shared import parse_args, sleep_awhile


# NOTE: Quota is ~400qpm, average single-process throughout is ~13qpm, so choosing 25
# processes to be safe on parallelization.
def enrich_people(emails, process_ct=25):
    payloads = _build_payloads(emails, "email")
    df = parallel_apply(payloads, _enrich_point, base_url="https://api.fullcontact.com/v3/person.enrich",
                        process_ct=process_ct, parallelization_module="gevent")
    return df

def enrich_companies(domains, process_ct=25):
    payloads = _build_payloads(domains, "domain")
    df = parallel_apply(payloads, _enrich_point, base_url="https://api.fullcontact.com/v3/company.enrich",
                        process_ct=process_ct, parallelization_module="gevent")
    return df

def _build_payloads(values, key):
    index = values.index if isinstance(values, (pd.Series, pd.DataFrame)) else range(len(values))
    payloads = pd.Series([{ key: value } for value in values], index=index)
    return payloads

@cache_today
def _enrich_point(payload, base_url):
    result = {}
    try:
        request = urllib.request.Request(base_url)
        request.add_header("Authorization", f"Bearer { os.environ['FULL_CONTACT_KEY'] }")

        logging.debug("Trying to enrich: %s: %s", base_url, payload)
        response = urllib.request.urlopen(request, json.dumps(payload).encode("utf-8"), timeout=10.000)
        result = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as error:
        if error.code == 404:
            logging.debug("Couldn't find entity: %s", payload)
        else:
            logging.warn("Couldn't enrich entity: %s: %s", base_url, payload, exc_info=True)
    except:
        logging.warn("Couldn't enrich entity: %s: %s", base_url, payload, exc_info=True)

    return result

if __name__ == "__main__":
    args = parse_args("Enrich people from FullContact", [
        dict(name_or_flags="emails", nargs="+", help="emails to look up in FullContact"),
    ])

    df = enrich_people(args.emails)
    import pdb; pdb.set_trace()

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


# NOTE: Quota is ~400qpm, average single-call throughout is ~60qpm, so choosing 5
# processes to be safe on parallelization.
def enrich_people(emails, process_ct=5):
    payloads = [{ "email": email } for email in emails]
    df = parallel_apply(payloads, _enrich_point, process_ct=process_ct,
                        base_url="https://api.fullcontact.com/v3/person.enrich")
    return df

def enrich_companies(domains, process_ct=5):
    payloads = [{ "domain": domain } for domain in domains]
    df = parallel_apply(payloads, _enrich_point, process_ct=process_ct,
                        base_url="https://api.fullcontact.com/v3/company.enrich")
    return df

@cache_today
def _enrich_point(payload, base_url):
    result = {}
    try:
        request = urllib.request.Request(base_url)
        request.add_header("Authorization", f"Bearer { os.environ['FULL_CONTACT_KEY'] }")

        logging.debug("Trying to enrich: %s: %s", base_url, payload)
        response = urllib.request.urlopen(request, json.dumps(payload).encode("utf-8"))
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

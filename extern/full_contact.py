#! /usr/bin/env python

import json
import os
import urllib.request

import pandas as pd

from util.shared import parse_args


def enrich_people(emails):
    results = [_enrich_point(email=email) for email in emails]
    df = pd.DataFrame(results, index=emails)
    return df

def enrich_companies(domains):
    results = [_enrich_point(domain=domain) for domain in domains]
    df = pd.DataFrame(results, index=domains)
    return df

# https://docs.fullcontact.com/?python#person-enrichment
def enrich_person(email=None, twitter=None, full_name=None):
    payload = {}
    if email is not None:
        payload["email"] = email
    if twitter is not None:
        payload["twitter"] = twitter
    if full_name is not None:
        payload["fullName"] = full_name

def _enrich_point(**payload):
    request = urllib.request.Request("https://api.fullcontact.com/v3/person.enrich")
    request.add_header("Authorization", f"Bearer { os.environ['FULL_CONTACT_KEY'] }")

    response = urllib.request.urlopen(request, json.dumps(payload).encode("utf-8"))
    result = json.loads(response.read().decode("utf-8"))
    return result

if __name__ == "__main__":
    args = parse_args("Enrich people from FullContact", [
        dict(name_or_flags="emails", nargs="+", help="emails to look up in FullContact"),
    ])

    df = enrich_people(args.emails)
    import pdb; pdb.set_trace()

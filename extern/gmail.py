#! /usr/local/bin/python2

# FIXME: This currently only works in python2 due to weird library issues with google.
# For now, to hack around this, we're pushing those google imports into the method. If
# the results are already pre-cached, which you force by

import os.path
import cloudpickle
import pandas as pd

from util.caching import cache_today
from util.shared import parse_args

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def _query_gmail(query, pages_max=100, force=False):
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    def __build_service():
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = cloudpickle.load(token)

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=55542)
            with open('token.pickle', 'wb') as token:
                cloudpickle.dump(creds, token)

        service = build('gmail', 'v1', credentials=creds)
        return service

    service = __build_service()

    @cache_today
    def __search_gmail(force=False, **kwargs):
        return service.users().messages().list(**kwargs).execute()

    results = []
    params = {
        "q": query,
        "userId": "me",
    }
    for ix in range(pages_max):
        result = __search_gmail(force=force, **params)
        results += result["messages"]

        if "nextPageToken" in result:
            params["pageToken"] = result["nextPageToken"]
        else:
            break

    @cache_today
    def __fetch_details(row, force=False, format="metadata"):
        return service.users().messages().get(userId="me", id=row["id"], format=format).execute()

    search_df = pd.DataFrame(results)
    messages_df = search_df.apply(__fetch_details, axis=1, result_type="expand", force=force)
    messages_df["internalDate"] = pd.to_datetime(messages_df["internalDate"], unit="ms")

    def __extract_metadata(row):
        df = pd.DataFrame(row["payload"]["headers"]).set_index("name").sort_index()
        return {
            "sender": df.loc["From", "value"],
            "subject": df.loc["Subject", "value"],
            "sent_at": df.loc["Date", "value"],
        }
    metadata_df = messages_df.apply(__extract_metadata, axis=1, result_type="expand")
    metadata_df["sent_at"] = pd.to_datetime(metadata_df["sent_at"], utc=True).dt.tz_convert("US/Pacific")

    full_df = pd.concat([messages_df, metadata_df], axis=1)[["id", "sender", "subject", "sent_at", "snippet"]]
    return full_df

if __name__ == '__main__':
    args = parse_args("Download emails from Gmail", [
         dict(name_or_flags="--force", action="store_true", help="whether to forcibly avoid cache"),
         dict(name_or_flags="query", help="email query to use"),
    ])
    results_df = _query_gmail(args.query, force=args.force)
    import pdb; pdb.set_trace()

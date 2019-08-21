import json
import logging
import os
import time
import urllib

import boto3
import numpy as np
import pandas as pd
import regex as re

def _parse_transcription(handle, speaker_labels={}):
    if isinstance(handle, str):
        from util.aws.s3 import s3_download
        handle = open(s3_download(handle))
    results = json.load(handle)

    __to_timedelta = lambda series: pd.to_timedelta(series.astype(float), unit="S").dt.round("S")

    transcript_df = pd.DataFrame(results["results"]["items"])
    transcript_df["start_time"] = __to_timedelta(transcript_df["start_time"])
    transcript_df["end_time"] = __to_timedelta(transcript_df["end_time"])
    transcript_df["content"] = transcript_df["alternatives"].apply(lambda row: row[0]["content"])

    transcript_df["confidence"] = transcript_df["alternatives"].apply(lambda row: row[0]["confidence"]).astype(float)
    transcript_df["confidence"] = (transcript_df["confidence"] + 1e-5).clip(upper=1.000)
    transcript_df["confidence"] = np.log(transcript_df["confidence"])

    if "speaker_labels" in results["results"]:
        speaker_df = pd.DataFrame(results["results"]["speaker_labels"]["segments"])
        speaker_df["start_time"] = __to_timedelta(speaker_df["start_time"])
        speaker_df["end_time"] = __to_timedelta(speaker_df["end_time"])

        transcript_df.set_index("start_time", inplace=True)
        def __content(row):
            mask = (row["start_time"] <= transcript_df.index) & (transcript_df.index < row["end_time"])
            content = " ".join(transcript_df.loc[mask, "content"])
            confidence = transcript_df.loc[mask, "confidence"].sum()
            return content, confidence
        speaker_df[["content", "confidence"]] = speaker_df.apply(__content, axis=1, result_type="expand")

        speaker_df.rename(columns={ "speaker_label": "speaker" }, inplace=True)
        if speaker_labels:
            speaker_df["speaker"] = speaker_df["speaker"].apply(speaker_labels.get)

        transcript_df = speaker_df[["speaker", "content", "confidence", "start_time", "end_time"]]

    return transcript_df

def _transcribe_audio(s3_target, s3_source, name=None, speaker_ct=2,
                      language="en-US", region="us-west-1", retries=10):
    client = boto3.client("transcribe")

    job_name = name or re.sub(r"\W", "_", s3_source)
    s3_components = urllib.parse.urlparse(s3_source)
    client.start_transcription_job(**{
        "TranscriptionJobName": job_name,
        "LanguageCode": language,
        "MediaFormat": os.path.splitext(s3_components.path)[-1][1:],
        "Media": {
            "MediaFileUri": f"https://s3-{ region }.amazon.aws.com/{ s3_components.netloc }/{ s3_components.path }",
        },
        "OutputBucketName": urllib.parse.urlparse(s3_source).netloc,
        "Settings": {
            "ShowSpeakerLabels": True,
            "MaxSpeakerLabels": speaker_ct,
        }
    })

    for ix in range(retries):
        job = client.get_transcription_job(TranscriptionJobName=job_name)
        if job["TranscriptionJobStatus"] != "IN_PROGRESS":
            break

        sleep_s = 2.000 ** ix
        logging.warn("Retrying %s after %.3f seconds", job_name, sleep_s)
        time.sleep(sleep_s)

    # FIXME: Move this to s3_target instead of overwriting it.
    s3_interim_path = job.get("Transcript", {}).get("TranscriptFileUri")
    if job["TranscriptionJobStatus"] != "COMPLETED":
        logging.error("Couldn't complete %s job: %s [%s]: %s", job_name, job["TranscriptionJobStatus"],
                      job.get("FailureReason"), s3_interim_path)
        return None

    transcript_df = _parse_transcription(s3_interim_path)
    return transcript_df

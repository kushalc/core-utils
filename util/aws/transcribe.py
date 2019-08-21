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

def _transcribe_audio(s3_target_path, s3_source_path, name=None, speaker_ct=2,
                      language="en-US", region="us-west-1", retries=10):
    transcribe_client = boto3.client("transcribe")

    job_name = name or re.sub(r"\W", "_", s3_target_path)
    s3_source_cmps = urllib.parse.urlparse(s3_source_path)
    s3_target_cmps = urllib.parse.urlparse(s3_target_path)
    transcribe_client.start_transcription_job(**{
        "TranscriptionJobName": job_name,
        "LanguageCode": language,
        "MediaFormat": os.path.splitext(s3_source_cmps.path)[-1][1:],
        "Media": {
            "MediaFileUri": s3_source_path,
        },
        "OutputBucketName": s3_target_cmps.netloc,
        "Settings": {
            "ShowSpeakerLabels": True,
            "MaxSpeakerLabels": speaker_ct,
        }
    })

    assert(retries >= 0)
    for ix in range(retries + 1):
        job = transcribe_client.get_transcription_job(TranscriptionJobName=job_name).get("TranscriptionJob", {})
        if job.get("TranscriptionJobStatus") != "IN_PROGRESS":
            logging.info("Stopping %s job: %s", job_name, job)
            break

        sleep_s = 2.000 ** ix
        logging.debug("Retrying %s job after %.0f seconds", job_name, sleep_s)
        time.sleep(sleep_s)

    s3_interim_path = re.sub(r"https://s3\..*\.amazonaws\.com/", "s3://", job.get("Transcript", {}).get("TranscriptFileUri"))
    s3_interim_cmps = urllib.parse.urlparse(s3_interim_path)
    if job["TranscriptionJobStatus"] != "COMPLETED":
        logging.error("Couldn't complete %s job: %s [%s]: %s", job_name, job["TranscriptionJobStatus"],
                      job.get("FailureReason"), s3_interim_path)
        return None

    s3_client = boto3.client("s3")
    s3_client.copy_object(**{
        "CopySource": {
            "Bucket": s3_interim_cmps.netloc,
            "Key": s3_interim_cmps.path.lstrip("/"),
        },

        "Bucket": s3_target_cmps.netloc,
        "Key": s3_target_cmps.path.lstrip("/"),
    })

    s3_client.delete_object(Bucket=s3_interim_cmps.netloc, Key=s3_interim_cmps.path.lstrip("/"))

    transcript_df = _parse_transcription(s3_target_path)
    return transcript_df

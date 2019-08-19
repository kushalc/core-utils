import json
import numpy as np
import pandas as pd

def _parse_transcription(handle, speaker_labels={}):
    results = json.load(handle)

    transcript_df = pd.DataFrame(results["results"]["items"])
    transcript_df["start_time"] = transcript_df["start_time"].astype(float)
    transcript_df["end_time"] = transcript_df["end_time"].astype(float)
    transcript_df["content"] = transcript_df["alternatives"].apply(lambda row: row[0]["content"])

    transcript_df["confidence"] = transcript_df["alternatives"].apply(lambda row: row[0]["confidence"]).astype(float)
    transcript_df["confidence"] = (transcript_df["confidence"] + 1e-5).clip(upper=1.000)
    transcript_df["confidence"] = np.log(transcript_df["confidence"])

    if "speaker_labels" in results["results"]:
        speaker_df = pd.DataFrame(results["results"]["speaker_labels"]["segments"])
        speaker_df["start_time"] = speaker_df["start_time"].astype(float)
        speaker_df["end_time"] = speaker_df["end_time"].astype(float)

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

#!/usr/bin/env python

import datetime
import logging
import os
from urllib.parse import urlparse, urlunparse

import boto3


class TimeoutException(BaseException):
    def __init__(self, frame=None, *args, **kwargs):
        self.stack = traceback.extract_stack(frame)

def get_env(environment=os.environ.get("DJANGO_ENV", "development")):
    name = environment if environment == "production" \
                       else "%s-%s" % (environment, os.environ.get("USER"))
    return name

# FIXME: s3_path and scoped_path are old and crufty and need some love.
INTERNAL_BUCKET = "talentworks-data"
CUSTOMER_BUCKET = "talentworks-uploads"
def s3_path(path, bucket=INTERNAL_BUCKET, **kwargs):
    if not isinstance(path, str):
        path = scoped_path(path, **kwargs)
    return urlunparse(("s3", bucket, path, None, None, None))

def scoped_path(path, prefix=None, dt=datetime.datetime.now(), request_id=None, user_id=None):
    # NOTE: path may have None __variant__
    path = [cp.lstrip("/") for cp in path if cp is not None]
    components = [prefix, get_env(), dt.strftime("%Y-%m-%d") if dt else None, user_id, request_id] + path
    path = os.path.join(*list(map(str, [_f for _f in components if _f])))
    return path

def parse_s3(raw_path):
    parsed = urlparse(raw_path)
    s3_bucket = parsed.netloc
    s3_path = parsed.path.lstrip("/")
    return s3_bucket, s3_path

BOTO_CLIENT = boto3.client("s3")
def s3_exists(path, **kwargs):
    exists = False

    s3_bucket, s3_path = parse_s3(path)
    if os.path.exists(_cached_s3_path(s3_bucket, s3_path)):
        exists = True
    elif s3_bucket:
        results = BOTO_CLIENT.list_objects_v2(Bucket=s3_bucket, Prefix=s3_path)
        matches = set(r["Key"] for r in results.get("Contents", []))
        exists = s3_path in matches

    if not exists:
        exists = os.path.exists(path)

    return exists

BOTO_S3 = boto3.resource("s3")
def s3_download(source, dest=None, **kwargs):
    s3_bucket, s3_path = parse_s3(source)
    if not dest:
        dest = _cached_s3_path(s3_bucket, s3_path)

    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    BOTO_S3.Bucket(s3_bucket).download_file(s3_path, dest)
    return dest

def _cached_s3_path(s3_bucket, s3_path):
    return os.path.join(os.environ.get("APP_ROOT", "."), "tmp", s3_bucket, s3_path)

def s3_upload(source, dest):
    with open(source, "rb") as handle:
        s3_bucket, s3_path = parse_s3(dest)
        BOTO_S3.Bucket(s3_bucket).put_object(Key=s3_path, Body=handle)

def s3_delete(path):
    s3_bucket, s3_path = parse_s3(path)
    BOTO_CLIENT.delete_object(Bucket=s3_bucket, Key=s3_path)

def s3_presign(raw_url, expires_in=3600):
    bucket, key = parse_s3(raw_url)
    presigned_url = BOTO_CLIENT.generate_presigned_url(
        ClientMethod='get_object',
        Params={
            'Bucket': bucket,
            'Key': key,
        }
    )
    return presigned_url

def s3_glob(prefix, regex=None, bucket=INTERNAL_BUCKET):
    try:
        s3_bucket = get_bucket(bucket)
        for s3_object in s3_bucket.list(prefix=prefix):
            if regex and not regex.match(s3_object.key):
                continue
            yield s3_path(s3_object.key, bucket)
    except:
        logging.warning("Couldn't glob S3: %s, %s", prefix, regex,
                        exc_info=True)

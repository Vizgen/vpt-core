import json
import os
import warnings
from enum import Enum
from typing import Tuple, Callable

import gcsfs
import rasterio.session
import tenacity
from fsspec import AbstractFileSystem, FSTimeoutError
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from rasterio.errors import NotGeoreferencedWarning
from s3fs import S3FileSystem
from tenacity import (
    stop_after_delay,
    stop_after_attempt,
    retry_if_exception_type,
    wait_fixed,
)

from vpt_core import (
    AWS_ACCESS_KEY_VAR,
    AWS_PROFILE_NAME_VAR,
    AWS_SECRET_KEY_VAR,
    GCS_SERVICE_ACCOUNT_KEY_VAR,
    log,
)


class Protocol(Enum):
    LOCAL = 0
    S3 = 1
    GCS = 2


AWS_PROFILE_NAME = None
AWS_ACCESS_KEY = None
AWS_SECRET_KEY = None

GCS_SERVICE_ACCOUNT_KEY = None


def initialize_filesystem(
    aws_access_key=None,
    aws_secret_key=None,
    aws_profile_name=None,
    gcs_service_account_key=None,
):
    global AWS_PROFILE_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY, GCS_SERVICE_ACCOUNT_KEY

    if all([aws_profile_name, aws_access_key, aws_secret_key]):
        log.warning("Both AWS profile name and access key/secret key pair are specified.")

    AWS_PROFILE_NAME = aws_profile_name if aws_profile_name else os.environ.get(AWS_PROFILE_NAME_VAR)
    AWS_ACCESS_KEY = aws_access_key if aws_access_key else os.environ.get(AWS_ACCESS_KEY_VAR)
    AWS_SECRET_KEY = aws_secret_key if aws_secret_key else os.environ.get(AWS_SECRET_KEY_VAR)

    if gcs_service_account_key:
        GCS_SERVICE_ACCOUNT_KEY = gcs_service_account_key
    else:
        GCS_SERVICE_ACCOUNT_KEY = os.environ.get(GCS_SERVICE_ACCOUNT_KEY_VAR)
        try:
            GCS_SERVICE_ACCOUNT_KEY = json.loads(GCS_SERVICE_ACCOUNT_KEY)
        except Exception:
            pass


def filesystem_for_protocol(protocol: Protocol) -> AbstractFileSystem:
    global AWS_PROFILE_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY, GCS_SERVICE_ACCOUNT_KEY

    if protocol == Protocol.LOCAL:
        return LocalFileSystem()
    elif protocol == Protocol.S3:
        return S3FileSystem(key=AWS_ACCESS_KEY, secret=AWS_SECRET_KEY, profile=AWS_PROFILE_NAME)
    elif protocol == Protocol.GCS:
        return GCSFileSystem(token=GCS_SERVICE_ACCOUNT_KEY)

    raise NotImplementedError(f"Protocol {protocol} is not supported")


def prefix_for_protocol(protocol: Protocol) -> str:
    if protocol == Protocol.LOCAL:
        return ""
    elif protocol == Protocol.S3:
        return "s3://"
    elif protocol == Protocol.GCS:
        return "gcs://"
    raise NotImplementedError(f"Protocol {protocol} is not supported")


def protocol_path_split(uri: str) -> Tuple[Protocol, str]:
    if uri.startswith("s3://"):
        split = Protocol.S3, uri[5:]
    elif uri.startswith("gcs://"):
        split = Protocol.GCS, uri[6:]
    elif uri.startswith("gs://"):
        split = Protocol.GCS, uri[5:]
    else:
        split = Protocol.LOCAL, uri
    check_access_to_uri(filesystem_for_protocol(split[0]), split[1])
    return split


def check_access_to_uri(fs: AbstractFileSystem, path: str):
    try:
        fs.exists(path)
    except gcsfs.retry.HttpError as e:
        raise ValueError(
            f'{e.message}. Pass credentials or run the command "gcloud auth application-default login" '
            f"to authenticate"
        )


def filesystem_path_split(uri: str) -> Tuple[AbstractFileSystem, str]:
    protocol, path = protocol_path_split(uri)
    fs = filesystem_for_protocol(protocol)

    assert fs is not None

    check_access_to_uri(fs, path)

    return fs, path


def retrying_attempts():
    return tenacity.Retrying(
        stop=(stop_after_delay(60) | stop_after_attempt(5)),
        retry=retry_if_exception_type(FSTimeoutError),
        wait=wait_fixed(10),
    )


def vzg_open(uri: str, mode: str, **kwargs):
    protocol, path = protocol_path_split(uri)
    fs = filesystem_for_protocol(protocol)

    assert fs is not None

    return fs.open(path, mode, **kwargs)


def io_with_retries(uri: str, mode: str, callback: Callable, **kwargs):
    for attempt in retrying_attempts():
        with attempt, vzg_open(uri, mode, **kwargs) as f:
            return callback(f)


def __get_rasterio_session(uri: str):
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    global AWS_PROFILE_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY, GCS_SERVICE_ACCOUNT_KEY

    protocol, _ = protocol_path_split(uri)

    if protocol == Protocol.LOCAL:
        return rasterio.session.DummySession()
    elif protocol == Protocol.S3:
        return rasterio.session.AWSSession(
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            profile_name=AWS_PROFILE_NAME,
        )
    elif protocol == Protocol.GCS:
        return rasterio.session.GSSession(google_application_credentials=GCS_SERVICE_ACCOUNT_KEY)

    raise NotImplementedError(f"Protocol {protocol} is not supported")


def rasterio_open(uri: str):
    protocol, path = protocol_path_split(uri)

    prefix = "/vsigs/" if protocol == Protocol.GCS else prefix_for_protocol(protocol)
    rasterio_path = prefix + path

    return rasterio.open(rasterio_path)


def get_rasterio_environment(uri: str, gdal_cache_size: int = 512000000) -> rasterio.Env:
    return rasterio.Env(__get_rasterio_session(uri), GDAL_CACHEMAX=gdal_cache_size)


def get_storage_options(uri: str) -> dict:
    protocol, _ = protocol_path_split(uri)

    if protocol == Protocol.LOCAL:
        return dict()
    elif protocol == Protocol.S3:
        return dict(profile=AWS_PROFILE_NAME, key=AWS_ACCESS_KEY, secret=AWS_SECRET_KEY)
    elif protocol == Protocol.GCS:
        return dict(token=GCS_SERVICE_ACCOUNT_KEY)

    raise NotImplementedError()

import io
from collections.abc import Iterable, Iterator

import boto3


def get_wet_file_paths(max_files: int, cc_main_id: str) -> list[str]:
    """Get WET file paths from multiple segments
    S3 directory structure:
    - Each month contains 100 segments.
    - Each segment contains 900 WET files
    - total 900 * 100 = 90k WET files.
    one segment contains 900 .gz WET files.
    ex) s3://commoncrawl/crawl-data/CC-MAIN-2024-22/segments/
        - 1715971057216.39/
            - warc/
            - wet/
                -   CC-MAIN-20240517233122-20240518023122-00899.warc.wet.gz
                - ...
        - other segments ID/

    """

    s3_client = boto3.client("s3")

    bucket = "commoncrawl"
    # List ALL segments, not just one
    prefix = f"crawl-data/{cc_main_id}/segments/"

    file_paths = []
    paginator = s3_client.get_paginator("list_objects_v2")

    # Iterate through segments
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        if "CommonPrefixes" not in page:
            continue

        for segment in page["CommonPrefixes"]:
            segment_prefix = segment["Prefix"] + "wet/"

            # List WET files in this segment. default MaxKeys=1000 > 900 WET files per segment.
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=segment_prefix,
            )

            if "Contents" in response:
                for obj in response["Contents"]:
                    if obj["Key"].endswith(".warc.wet.gz"):
                        file_paths.append(f"s3://{bucket}/{obj['Key']}")

                        if len(file_paths) >= max_files:
                            print(f"Collected {len(file_paths)} WET file paths")
                            return file_paths

    print(f"Collected {len(file_paths)} WET file paths")
    return file_paths


def parse_wet_record_v2(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """
    Args:
    - lines: Iterable. Caller's each row.value is a str. But lines receives the generator of many lines.

    Retuns:
    - Iterator[tuple[str, str]

    Parsing based on WET file format logic:
    - From WARC-Target-URI:, fetch URL. lines without URL is skipped.
    - While URL is set, when blank line is found, flag=True for IN_CONTENT.
    - While URL is set and IN_CONTENT is True, then accumulate contents.
    - when WARC-Type: keyword is found, reset

    ex) aws s3 cp s3://commoncrawl/crawl-data/CC-MAIN-2024-22/segments/1715971057216.39/wet
        /CC-MAIN-20240517233122-20240518023122-00001.warc.wet.gz - | gunzip | head -100

    WARC/1.0
    WARC-Type: warcinfo                                                    <- record 1 (metadata, no URL)
    WARC-Date: 2024-05-31T01:24:21Z
    WARC-Filename: CC-MAIN-20240517233122-20240518023122-00001.warc.wet.gz
    WARC-Record-ID: <urn:uuid:be47515c-d86b-4ce6-82f6-8c3c11086d3c>
    Content-Type: application/warc-fields
    Content-Length: 368

    Software-Info: ia-web-commons.1.1.10-SNAPSHOT-20240513074037
    Extracted-Date: Fri, 31 May 2024 01:24:21 GMT
    robots: checked via crawler-commons 1.5-SNAPSHOT (https://github.com/crawler-commons/crawler-commons)
    isPartOf: CC-MAIN-2024-22
    operator: Common Crawl Admin (info@commoncrawl.org)
    description: Wide crawl of the web for May 2024
    publisher: Common Crawl



    WARC/1.0
    WARC-Type: conversion
    WARC-Target-URI: http://0-50.ru/news/line/2013-03-26/id_30926.html    <- record 2 (get URL)
    WARC-Date: 2024-05-18T01:05:37Z
    WARC-Record-ID: <urn:uuid:6e69bc67-a141-4cc8-b949-a3a9b647d87c>
    WARC-Refers-To: <urn:uuid:a25e1f4e-42d7-4e2f-8a77-0f8645af7b2c>
    WARC-Block-Digest: sha1:FCFS6CEWGSGDH4JZJLWRSXOKZ4JUFI52
    WARC-Identified-Content-Language: rus
    Content-Type: text/plain
    Content-Length: 17217
                                                                          <- blank line = content starts
    (document contents start)
    """

    current_url = None
    content_buffer = io.StringIO()
    in_content = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("WARC-Type:"):
            # WARC-Type is the start of new doc and finalize previous record
            if current_url and content_buffer.tell() > 0:
                content_buffer.seek(0)
                text = content_buffer.read().strip()
                if len(text) > 100:  # Filter short content
                    # With yield, this function becomes a generator (each record
                    # flows downstream immediately, then gets garbage collected).
                    # We use mapPartition as lazy eval.
                    yield (current_url, text)

            # Reset state
            current_url = None
            content_buffer.seek(0)
            content_buffer.truncate(0)
            in_content = False

        elif stripped.startswith("WARC-Target-URI:"):
            # Extract URL
            current_url = stripped[17:].strip()  # Skip "WARC-Target-URI: "

        elif stripped == "" and current_url:
            # Blank line after headers means content starts next
            in_content = True

        elif in_content and stripped:
            # it means we are at contents lines - use StringIO for efficient text accumulation
            content_buffer.write(stripped)
            content_buffer.write(" ")

    # Handle last record
    if current_url and content_buffer.tell() > 0:
        content_buffer.seek(0)
        text = content_buffer.read().strip()
        if len(text) > 100:
            yield (current_url, text)

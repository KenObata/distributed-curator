"""Unit tests for test/integration_test/wet_file_utils.py — WET file parsing and S3 listing."""

from unittest.mock import MagicMock, patch

from wet_file_utils import get_wet_file_paths, parse_wet_record_v2


class TestParseWetRecordV2:
    """Tests for the WET record streaming parser."""

    def test_single_record(self):
        """Parse a single complete WET record."""
        lines = [
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.com/page1",
            "Content-Type: text/plain",
            "Content-Length: 200",
            "",
            "This is the extracted text content from page one. " * 5,
        ]
        records = list(parse_wet_record_v2(lines))  # (current_url, text)

        assert len(records) == 1
        assert records[0][0] == "http://example.com/page1"
        assert "extracted text content" in records[0][1]

    def test_multiple_records(self):
        """Parse two consecutive WET records."""
        lines = [
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.com/page1",
            "Content-Length: 200",
            "",
            "First page content that is long enough to pass the filter threshold easily. " * 3,
            "",
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.com/page2",
            "Content-Length: 300",
            "",
            "Second page content that is also long enough to pass the filter threshold. " * 3,
        ]
        records = list(parse_wet_record_v2(lines))

        assert len(records) == 2
        assert records[0][0] == "http://example.com/page1"
        assert records[1][0] == "http://example.com/page2"

    def test_warcinfo_record_skipped(self):
        """The warcinfo metadata record (no URL) should be skipped."""
        lines = [
            "WARC/1.0",
            "WARC-Type: warcinfo",
            "WARC-Date: 2024-05-31T01:16:46Z",
            "Content-Type: application/warc-fields",
            "Content-Length: 368",
            "",
            "Software-Info: ia-web-commons.1.1.10-SNAPSHOT",
            "isPartOf: CC-MAIN-2024-22",
            "operator: Common Crawl Admin",
            "",
            "",
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.com/real-page",
            "Content-Length: 500",
            "",
            "This is actual page content that should be captured by the parser. " * 3,
        ]
        records = list(parse_wet_record_v2(lines))

        assert len(records) == 1
        assert records[0][0] == "http://example.com/real-page"

    def test_short_content_filtered(self):
        """Records with text <= 100 chars should be filtered out."""
        lines = [
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.com/short",
            "Content-Length: 10",
            "",
            "Too short.",
        ]
        records = list(parse_wet_record_v2(lines))

        assert len(records) == 0

    def test_exactly_100_chars_filtered(self):
        """Record with exactly 100 chars should be filtered (> 100, not >=)."""
        text_100 = "x" * 100
        lines = [
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.com/boundary",
            "Content-Length: 100",
            "",
            text_100,
        ]
        records = list(parse_wet_record_v2(lines))

        assert len(records) == 0

    def test_101_chars_passes(self):
        """Record with 101 chars should pass the filter."""
        text_101 = "x" * 101
        lines = [
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.com/boundary",
            "Content-Length: 101",
            "",
            text_101,
        ]
        records = list(parse_wet_record_v2(lines))

        assert len(records) == 1

    def test_multiline_content_joined(self):
        """Multiple content lines should be joined with spaces."""
        lines = [
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.com/multi",
            "Content-Length: 500",
            "",
            "Line one of content.",
            "Line two of content.",
            "Line three of content that makes this long enough to pass the hundred char filter.",
        ]
        records = list(parse_wet_record_v2(lines))

        assert len(records) == 1
        assert "Line one of content. Line two of content. Line three" in records[0][1]

    def test_blank_lines_in_content_ignored(self):
        """Blank lines within content section should not break parsing.
        The parser only writes non-empty stripped lines to the buffer,
        so blank lines in content are simply skipped."""
        lines = [
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.com/blanks",
            "Content-Length: 500",
            "",
            "First paragraph of content that needs to be quite long to pass the filter threshold.",
            "",
            "Second paragraph after a blank line within the same record's content section.",
        ]
        records = list(parse_wet_record_v2(lines))

        assert len(records) == 1
        # Both paragraphs captured — blank line in between was just skipped
        assert "First paragraph" in records[0][1]
        assert "Second paragraph" in records[0][1]

    def test_last_record_yielded(self):
        """The final record in the file (no trailing WARC-Type:) should be yielded."""
        lines = [
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.com/last",
            "Content-Length: 300",
            "",
            "This is the very last record in the file with no following WARC-Type marker. " * 3,
        ]
        records = list(parse_wet_record_v2(lines))

        assert len(records) == 1
        assert records[0][0] == "http://example.com/last"

    def test_generator_behavior(self):
        """Verify it's a true generator, not a list — records stream lazily."""
        lines = [
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.com/page1",
            "",
            "Content long enough to pass the hundred character filter threshold for this test. " * 2,
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.com/page2",
            "",
            "More content that is also long enough to pass the hundred character filter threshold. " * 2,
        ]
        result = parse_wet_record_v2(lines)

        # Should be a generator, not a list
        import types

        assert isinstance(result, types.GeneratorType)

        # Pull records one at a time
        first = next(result)
        assert first[0] == "http://example.com/page1"

        second = next(result)
        assert second[0] == "http://example.com/page2"

    def test_empty_input(self):
        """Empty input should yield no records."""
        records = list(parse_wet_record_v2([]))

        assert len(records) == 0

    def test_url_extraction_strips_whitespace(self):
        """URL should be stripped of leading/trailing whitespace."""
        lines = [
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI:   http://example.com/spaces   ",
            "Content-Length: 300",
            "",
            "Content that is long enough to pass the one hundred character filter for this test case. " * 2,
        ]
        records = list(parse_wet_record_v2(lines))

        assert len(records) == 1
        assert records[0][0] == "http://example.com/spaces"

    def test_realistic_wet_format(self):
        """Test with a realistic WET file structure including warcinfo + two records."""
        lines = [
            "WARC/1.0",
            "WARC-Type: warcinfo",
            "WARC-Date: 2024-05-31T01:24:21Z",
            "WARC-Filename: CC-MAIN-20240517233122-20240518023122-00001.warc.wet.gz",
            "WARC-Record-ID: <urn:uuid:be47515c-d86b-4ce6-82f6-8c3c11086d3c>",
            "Content-Type: application/warc-fields",
            "Content-Length: 368",
            "",
            "Software-Info: ia-web-commons.1.1.10-SNAPSHOT-20240513074037",
            "Extracted-Date: Fri, 31 May 2024 01:24:21 GMT",
            "robots: checked via crawler-commons 1.5-SNAPSHOT",
            "isPartOf: CC-MAIN-2024-22",
            "operator: Common Crawl Admin (info@commoncrawl.org)",
            "description: Wide crawl of the web for May 2024",
            "publisher: Common Crawl",
            "",
            "",
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://0-50.ru/news/line/2013-03-26/id_30926.html",
            "WARC-Date: 2024-05-18T01:05:37Z",
            "WARC-Record-ID: <urn:uuid:6e69bc67-a141-4cc8-b949-a3a9b647d87c>",
            "WARC-Refers-To: <urn:uuid:a25e1f4e-42d7-4e2f-8a77-0f8645af7b2c>",
            "WARC-Block-Digest: sha1:FCFS6CEWGSGDH4JZJLWRSXOKZ4JUFI52",
            "WARC-Identified-Content-Language: rus",
            "Content-Type: text/plain",
            "Content-Length: 17217",
            "",
            "This is Russian news article content from 0-50.ru about some important event. " * 3,
            "More details about the event with additional context and information. " * 2,
            "",
            "WARC/1.0",
            "WARC-Type: conversion",
            "WARC-Target-URI: http://example.org/second-article",
            "WARC-Date: 2024-05-18T01:06:00Z",
            "Content-Type: text/plain",
            "Content-Length: 5000",
            "",
            "English article content with enough text to pass the length filter easily. " * 4,
        ]
        records = list(parse_wet_record_v2(lines))

        assert len(records) == 2
        assert records[0][0] == "http://0-50.ru/news/line/2013-03-26/id_30926.html"
        assert records[1][0] == "http://example.org/second-article"
        assert "Russian news article" in records[0][1]
        assert "English article" in records[1][1]


class TestGetWetFilePaths:
    """Tests for S3 listing of WET file paths."""

    @patch("wet_file_utils.boto3")
    def test_collects_paths_up_to_max(self, mock_boto3):
        """Should stop collecting once max_files is reached."""
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        # Mock paginator for segment listing
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "crawl-data/CC-MAIN-2024-22/segments/seg1/"},
                    {"Prefix": "crawl-data/CC-MAIN-2024-22/segments/seg2/"},
                ]
            }
        ]

        # Mock list_objects_v2 for WET files in each segment
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"crawl-data/CC-MAIN-2024-22/segments/seg1/wet/file-{i:05d}.warc.wet.gz"} for i in range(900)
            ]
        }

        paths = get_wet_file_paths(max_files=5, cc_main_id="CC-MAIN-2024-22")

        assert len(paths) == 5
        assert all(p.startswith("s3://commoncrawl/") for p in paths)
        assert all(p.endswith(".warc.wet.gz") for p in paths)

    @patch("wet_file_utils.boto3")
    def test_skips_non_wet_files(self, mock_boto3):
        """Should only collect .warc.wet.gz files, not other file types."""
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "crawl-data/CC-MAIN-2024-22/segments/seg1/"},
                ]
            }
        ]

        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "segments/seg1/wet/file-00000.warc.wet.gz"},
                {"Key": "segments/seg1/wet/file-00001.warc.gz"},  # not a WET file
                {"Key": "segments/seg1/wet/file-00002.txt"},  # not a WET file
                {"Key": "segments/seg1/wet/file-00003.warc.wet.gz"},
            ]
        }

        paths = get_wet_file_paths(max_files=100, cc_main_id="CC-MAIN-2024-22")

        assert len(paths) == 2

    @patch("wet_file_utils.boto3")
    def test_multiple_segments(self, mock_boto3):
        """Should iterate through multiple segments."""
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "crawl-data/CC-MAIN-2024-22/segments/seg1/"},
                    {"Prefix": "crawl-data/CC-MAIN-2024-22/segments/seg2/"},
                ]
            }
        ]

        # Return 3 files per segment
        mock_s3.list_objects_v2.side_effect = [
            {"Contents": [{"Key": f"segments/seg1/wet/file-{i:05d}.warc.wet.gz"} for i in range(3)]},
            {"Contents": [{"Key": f"segments/seg2/wet/file-{i:05d}.warc.wet.gz"} for i in range(3)]},
        ]

        paths = get_wet_file_paths(max_files=100, cc_main_id="CC-MAIN-2024-22")

        assert len(paths) == 6

    @patch("wet_file_utils.boto3")
    def test_empty_segment(self, mock_boto3):
        """Should handle segments with no Contents gracefully."""
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "crawl-data/CC-MAIN-2024-22/segments/empty-seg/"},
                ]
            }
        ]

        mock_s3.list_objects_v2.return_value = {}  # no Contents key

        paths = get_wet_file_paths(max_files=100, cc_main_id="CC-MAIN-2024-22")

        assert len(paths) == 0

    @patch("wet_file_utils.boto3")
    def test_cc_main_id_used_in_prefix(self, mock_boto3):
        """Should use the cc_main_id parameter to construct the S3 prefix."""
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"CommonPrefixes": []}]

        get_wet_file_paths(max_files=10, cc_main_id="CC-MAIN-2025-01")

        mock_paginator.paginate.assert_called_once_with(
            Bucket="commoncrawl",
            Prefix="crawl-data/CC-MAIN-2025-01/segments/",
            Delimiter="/",
        )

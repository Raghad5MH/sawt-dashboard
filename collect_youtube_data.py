#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_youtube_data.py — pull Saudi podcast videos from the YouTube Data API v3.

This is the data-collection code of the project notebook, moved into its own
script. The dashboard (app.py) does NOT need it: it reads the ready-made file
data/clean_data.csv. Run this script only when you want to collect fresh data.

Usage
-----
    # 1) install the extra requirement
    pip install -r requirements-collector.txt

    # 2) give the API key through an environment variable (never write it in code)
    #    macOS / Linux :  export YOUTUBE_API_KEY="your-key"
    #    Windows (PS)  :  $env:YOUTUBE_API_KEY="your-key"

    # 3) collect
    python collect_youtube_data.py
    python collect_youtube_data.py --target 500 --output data/saudi_podcast_videos.csv

    # 4) optional: also write a notebook-style cleaned file for the dashboard
    python collect_youtube_data.py --clean-output data/clean_data_new.csv

Quota
-----
Every search page costs 100 quota units and the default daily quota is 10,000
units (about 100 search pages). If the quota runs out, the script stops
searching and keeps whatever it has already collected.

Existing files are never overwritten unless you pass --overwrite.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:  # the helpers below can be imported (and tested) without the Google library
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:  # pragma: no cover
    build = None

    class HttpError(Exception):  # type: ignore[no-redef]
        """Placeholder used when google-api-python-client is not installed."""


# --------------------------------------------------------------------------- #
# Settings copied from the notebook
# --------------------------------------------------------------------------- #
PODCASTS = [
    "ذا قال", "أسمار", "أسماري", "بترولي", "فراغ",
    "فنجان", "سقراط", "ثمانية", "سوالف بزنس",
    "مربع", "بدون ورق", "كنبة السبت", "جناية",
    "يصير خير", "متمم", "دواير", "محفوف",
    "بودكاست الغرفة", "بودكاست رفوف", "بودكاست أشياء",
    "بودكاست بترولي", "بودكاست أسمار", "بودكاست فراغ",
    "بودكاست سعودي", "بودكاست السعودية",
    "بودكاست حواري سعودي", "بودكاست مقابلات سعودية",
    "بودكاست رواد الأعمال السعودية",
    "بودكاست قصص سعودية", "بودكاست ثمانية",
    "بودكاست إذاعة مختلف",
]

PODCAST_MARKERS = [
    "بودكاست", "podcast",
    "فنجان", "سقراط", "أسمار", "أسماري",
    "ذا قال", "بترولي", "فراغ", "سوالف بزنس",
    "كنبة السبت", "بدون ورق", "مربع", "جناية",
    "محفوف", "دواير", "متمم", "يصير خير",
]

DEFAULT_TARGET = 1000
DEFAULT_MAX_PAGES = 2          # 50 results per page
DEFAULT_REGION = "SA"
DEFAULT_LANGUAGE = "ar"
DEFAULT_OUTPUT = Path("data") / "saudi_podcast.csv"


class QuotaExceeded(Exception):
    """Raised when the YouTube API daily quota is exhausted."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def relevant_title(title: str, markers=PODCAST_MARKERS) -> bool:
    """Keep only videos whose title mentions a podcast marker (same rule as the notebook)."""
    folded = (title or "").casefold()
    return any(word.casefold() in folded for word in markers)


_ISO = re.compile(r"^P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")


def iso_duration_to_seconds(value) -> float | None:
    """ISO 8601 duration such as 'PT3H3M9S' -> seconds (None if it cannot be read)."""
    if not isinstance(value, str):
        return None
    match = _ISO.match(value.strip())
    if not match or not any(match.groups()):
        return None
    days, hours, minutes, seconds = match.groups()
    return int(days or 0) * 86400 + int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0)


def seconds_to_readable(seconds: float | None) -> str | None:
    """Seconds -> 'H:MM:SS'."""
    if seconds is None or pd.isna(seconds):
        return None
    total = int(round(seconds))
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def is_quota_error(exc: Exception) -> bool:
    status = getattr(getattr(exc, "resp", None), "status", None)
    content = getattr(exc, "content", b"") or b""
    if isinstance(content, str):
        content = content.encode("utf-8", "ignore")
    return status == 403 and b"quota" in content.lower()


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def search_videos(
    youtube,
    podcasts=PODCASTS,
    target: int = DEFAULT_TARGET,
    max_pages: int = DEFAULT_MAX_PAGES,
    region: str = DEFAULT_REGION,
    language: str = DEFAULT_LANGUAGE,
    log=print,
):
    """Search YouTube for each podcast name. Returns (videos_by_id, errors)."""
    videos: dict[str, dict] = {}
    errors: list[dict] = []

    try:
        for podcast in podcasts:
            if len(videos) >= target:
                break
            for search_term in (f'"{podcast}"', f'"{podcast}" حلقة'):
                if len(videos) >= target:
                    break
                page_token = None
                for _ in range(max_pages):
                    if len(videos) >= target:
                        break
                    params = {
                        "part": "snippet",
                        "q": search_term,
                        "type": "video",
                        "maxResults": 50,
                        "regionCode": region,
                        "relevanceLanguage": language,
                        "order": "relevance",
                    }
                    if page_token:
                        params["pageToken"] = page_token
                    try:
                        response = youtube.search().list(**params).execute()
                    except HttpError as exc:
                        if is_quota_error(exc):
                            raise QuotaExceeded from exc
                        errors.append({"query": search_term, "error": str(exc)})
                        log(f"تعذر البحث عن: {search_term} | {exc}")
                        break

                    for item in response.get("items", []):
                        video_id = item.get("id", {}).get("videoId")
                        snippet = item.get("snippet", {})
                        title = snippet.get("title", "")
                        if not video_id or not relevant_title(title):
                            continue
                        videos[video_id] = {
                            "video_id": video_id,
                            "video_title": title,
                            "channel_name": snippet.get("channelTitle"),
                            "channel_id": snippet.get("channelId"),
                            "published_at": snippet.get("publishedAt"),
                            "matched_search": search_term,
                        }
                        if len(videos) >= target:
                            break

                    page_token = response.get("nextPageToken")
                    if not page_token:
                        break
            log(f"انتهى البحث عن {podcast} | المجموع: {len(videos)}")
    except QuotaExceeded:
        errors.append({"query": "(search)", "error": "YouTube API quota exceeded"})
        log("\nانتهت حصة YouTube API اليومية؛ سيتم الاحتفاظ بما جُمع حتى الآن.")

    return videos, errors


def fetch_video_details(youtube, videos: dict, log=print):
    """Fetch statistics and duration for the collected video ids. Returns (rows, errors)."""
    rows: list[dict] = []
    errors: list[dict] = []
    ids = list(videos.keys())

    for start in range(0, len(ids), 50):
        batch = ids[start:start + 50]
        try:
            response = youtube.videos().list(part="snippet,statistics,contentDetails", id=",".join(batch)).execute()
        except HttpError as exc:
            errors.append({"batch": start, "error": str(exc)})
            log(f"تعذر جلب إحصاءات الدفعة: {start} {exc}")
            if is_quota_error(exc):
                break
            continue

        for video in response.get("items", []):
            snippet = video.get("snippet", {})
            stats = video.get("statistics", {})
            video_id = video["id"]
            rows.append(
                {
                    **videos[video_id],
                    "channel_name": snippet.get("channelTitle"),
                    "published_at": snippet.get("publishedAt"),
                    "views": int(stats["viewCount"]) if "viewCount" in stats else None,
                    "likes": int(stats["likeCount"]) if "likeCount" in stats else None,
                    "comments": int(stats["commentCount"]) if "commentCount" in stats else None,
                    "duration": video.get("contentDetails", {}).get("duration"),
                    "video_url": f"https://www.youtube.com/watch?v={video_id}",
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    return rows, errors


def build_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["video_id"])
    return df.sort_values("views", ascending=False, na_position="last").reset_index(drop=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook-style cleaning: missing comments -> 0 and duration columns.

    The 30-minute rule is NOT applied here; the dashboard applies it and reports it.
    """
    out = df.copy()
    out["comments"] = out["comments"].fillna(0)
    seconds = out["duration"].map(iso_duration_to_seconds)
    out["duration_readable"] = seconds.map(seconds_to_readable)
    out["duration_seconds"] = seconds
    out["duration_minutes"] = seconds / 60
    return out


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Saudi podcast videos from the YouTube Data API v3.")
    p.add_argument("--api-key", default=None, help="API key (prefer the YOUTUBE_API_KEY environment variable).")
    p.add_argument("--target", type=int, default=DEFAULT_TARGET, help=f"Stop after this many relevant videos (default {DEFAULT_TARGET}).")
    p.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help=f"Result pages per query (default {DEFAULT_MAX_PAGES}, 50 results each).")
    p.add_argument("--region", default=DEFAULT_REGION, help=f"regionCode (default {DEFAULT_REGION}).")
    p.add_argument("--language", default=DEFAULT_LANGUAGE, help=f"relevanceLanguage (default {DEFAULT_LANGUAGE}).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Raw CSV output (default {DEFAULT_OUTPUT}).")
    p.add_argument("--clean-output", type=Path, default=None, help="Also write a notebook-style cleaned CSV here.")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    targets = [args.output] + ([args.clean_output] if args.clean_output else [])
    existing = [str(t) for t in targets if t.exists()]
    if existing and not args.overwrite:
        print("الملف موجود مسبقاً ولن يُستبدل:", ", ".join(existing))
        print("Output already exists. Pass --overwrite or choose another --output.")
        return 1

    api_key = args.api_key or os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("لم يُحدَّد مفتاح YouTube API. عيّن المتغير YOUTUBE_API_KEY أو استخدم --api-key.")
        print("No API key found. Set the YOUTUBE_API_KEY environment variable or use --api-key.")
        return 1
    if build is None:
        print("المكتبة google-api-python-client غير مثبتة: pip install -r requirements-collector.txt")
        return 1

    youtube = build("youtube", "v3", developerKey=api_key)
    videos, errors = search_videos(
        youtube, target=args.target, max_pages=args.max_pages, region=args.region, language=args.language
    )
    print("\nعدد معرفات الفيديوهات الفريدة:", len(videos))

    rows, detail_errors = fetch_video_details(youtube, videos)
    errors += detail_errors
    df = build_dataframe(rows)

    if df.empty:
        print("لم تُجمع نتائج. تحقق من API Key وحصة الاستخدام.")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nعدد المقاطع المحفوظة: {len(df)} | عدد القنوات: {df['channel_id'].nunique()} | الملف: {args.output}")

    if args.clean_output:
        args.clean_output.parent.mkdir(parents=True, exist_ok=True)
        clean_dataframe(df).to_csv(args.clean_output, index=False, encoding="utf-8-sig")
        print("الملف النظيف:", args.clean_output)

    if errors:
        print("عدد الأخطاء:", len(errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())

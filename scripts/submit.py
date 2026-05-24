"""POST submission.csv to the leaderboard.

Usage:
    export TML_API_KEY=<your key>
    python -m scripts.submit                              # uses submissions/submission.csv
    python -m scripts.submit --file path/to/other.csv
"""

import argparse
import os
import sys

import requests


BASE_URL = "http://34.63.153.158"
TASK_ID = "19-stolen-model-detection"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="submissions/submission.csv")
    p.add_argument("--api-key", default=os.environ.get("TML_API_KEY", ""))
    args = p.parse_args()

    if not args.api_key:
        print("error: set TML_API_KEY env var or pass --api-key", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(args.file):
        print(f"error: file not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    with open(args.file, "rb") as f:
        files = {"file": (os.path.basename(args.file), f, "csv")}
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": args.api_key},
            files=files,
            timeout=(10, 120),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    print(f"[http {resp.status_code}] {body}")
    if resp.status_code == 413:
        sys.exit(1)
    resp.raise_for_status()


if __name__ == "__main__":
    main()

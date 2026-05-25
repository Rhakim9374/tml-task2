"""Download suspect_models/suspect_NNN.safetensors for NNN in [start, end).

The pytorch docker image has no wget/curl, so we use stdlib urllib. Threading
gives us parallel downloads; 8 concurrent connections saturate HF's bandwidth
for a single worker without hammering it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import urllib.request

BASE = "https://huggingface.co/SprintML/tml26_task2/resolve/main"


def fetch(i: int, out_dir: str) -> tuple[str, str]:
    name = f"suspect_{i:03d}.safetensors"
    dst = os.path.join(out_dir, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return name, "skip"
    url = f"{BASE}/suspect_models/{name}"
    try:
        urllib.request.urlretrieve(url, dst)
        return name, "ok"
    except Exception as e:
        if os.path.exists(dst):
            os.remove(dst)
        return name, f"FAIL: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ids = range(args.start, args.end)

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch, i, args.out_dir): i for i in ids}
        for fut in concurrent.futures.as_completed(futures):
            name, status = fut.result()
            if status.startswith("FAIL"):
                failures.append(f"{name}: {status}")
                print(f"  {name}: {status}", file=sys.stderr, flush=True)
            else:
                print(f"  {name}: {status}", flush=True)

    if failures:
        print(f"\n{len(failures)} downloads failed:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)

    n = sum(1 for f in os.listdir(args.out_dir) if f.endswith(".safetensors"))
    print(f"downloaded {n} suspects to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

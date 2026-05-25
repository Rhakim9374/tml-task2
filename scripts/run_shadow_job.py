"""Dispatcher invoked by cluster/run_shadow_job.sh once per HTCondor Process.

Reads shadows/jobs.json, looks up the entry at --job-idx, runs its cmd.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-idx", type=int, required=True)
    ap.add_argument("--jobs-json", default="shadows/jobs.json")
    args = ap.parse_args()

    with open(args.jobs_json) as f:
        jobs = json.load(f)
    if args.job_idx < 0 or args.job_idx >= len(jobs):
        print(f"job-idx {args.job_idx} out of range (0..{len(jobs)-1})", file=sys.stderr)
        sys.exit(1)

    job = jobs[args.job_idx]
    print(f"[job {args.job_idx}] kind={job['kind']} label={job['label']} out={job['out']}", flush=True)
    print(f"[job {args.job_idx}] cmd={' '.join(job['cmd'])}", flush=True)

    result = subprocess.run(job["cmd"], check=False)
    if result.returncode != 0:
        print(f"[job {args.job_idx}] FAILED rc={result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"[job {args.job_idx}] OK", flush=True)


if __name__ == "__main__":
    main()

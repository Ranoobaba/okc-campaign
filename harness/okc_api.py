"""Thin client for the challenge's HTTP API, for agents and the loop.

Token: OKC_TOKEN env var, else the okc CLI's config (~/.config/okc/config.json or
~/Library/Application Support/okc/config.json). A person must run `okc login` once.

  python3 harness/okc_api.py me
  python3 harness/okc_api.py limits
  python3 harness/okc_api.py check  muse_bf16_proj problems/muse_bf16_proj/submission.py   # off the board
  python3 harness/okc_api.py submit muse_bf16_proj problems/muse_bf16_proj/submission.py   # scored
  python3 harness/okc_api.py status <job_id> [--wait]
  python3 harness/okc_api.py jobs
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://openkernelchallenge.dev/api/v1"
ROOT = Path(__file__).resolve().parent.parent
UA = "okc-campaign/0.1"


def token() -> str:
    t = os.environ.get("OKC_TOKEN")
    if t:
        return t
    f = ROOT / ".okc_token"
    if f.exists() and f.read_text().strip():
        return f.read_text().strip()
    for p in (Path.home() / "Library/Application Support/okc/config.json", Path.home() / ".config/okc/config.json"):
        if p.exists():
            try:
                cfg = json.loads(p.read_text())
                for k in ("token", "api_token"):
                    if cfg.get(k):
                        return cfg[k]
            except Exception:
                pass
    sys.exit("no token: run `okc login` (a person must accept the rules) or set OKC_TOKEN=$(okc api token)")


def call(method, path, body=None, auth=True):
    headers = {"User-Agent": UA, "Accept": "application/json", "Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {token()}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        raise SystemExit(f"HTTP {e.code} on {method} {path}: {msg[:800]}")


def submit(problem, path, mode):
    code = Path(path).read_text()
    first = code.splitlines()[0] if code else ""
    if not first.startswith("# okc: problem="):
        print(f"warning: first line is not '# okc: problem={problem}' (okc CLI needs it; the API does not)")
    r = call("POST", f"/problems/{problem}/submissions", {"code": code, "mode": mode})
    log = ROOT / "results" / "submissions.jsonl"
    log.parent.mkdir(exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "problem": problem, "file": str(path), "mode": mode, **r}) + "\n")
    return r


def status(job, wait=False):
    while True:
        r = call("GET", f"/jobs/{job}")
        st = r.get("status")
        if not wait or st in ("done", "failed", "rejected", "duplicate", "timeout", "error"):
            return r
        print(f"  {job}: {st} ...", file=sys.stderr)
        time.sleep(20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    if a.cmd == "me":
        out = call("GET", "/me")
    elif a.cmd == "limits":
        out = call("GET", "/problems")
    elif a.cmd in ("submit", "check"):
        problem, path = a.args
        out = submit(problem, path, "run" if a.cmd == "submit" else "check")
    elif a.cmd == "status":
        out = status(a.args[0], a.wait)
    elif a.cmd == "jobs":
        out = call("GET", "/jobs")
    elif a.cmd == "withdraw":
        out = call("POST", f"/jobs/{a.args[0]}/withdraw", {})
    else:
        sys.exit(f"unknown cmd {a.cmd}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

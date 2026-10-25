"""Leaderboard snapshot: fetch every board, save JSON under snapshots/, print the picture.

  python3 harness/board.py            # all boards
  python3 harness/board.py --team Ranoobaba
"""

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

BASE = "https://openkernelchallenge.dev/api/v1"
P = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"]
ROOT = Path(__file__).resolve().parent.parent


def get(path):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "okc-campaign/0.1 (curl-compatible)", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="Ranoobaba")
    a = ap.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    snap = {"at": stamp, "overall": get("/leaderboard"), "queue": get("/queue"), "boards": {}}
    for p in P:
        snap["boards"][p] = get(f"/leaderboard/{p}")
    (ROOT / "snapshots").mkdir(exist_ok=True)
    (ROOT / "snapshots" / f"{stamp}.json").write_text(json.dumps(snap))
    q = snap["queue"]
    print(f"queue: backlog {q['backlog']} running {q['running']} gpus {q['kernel_gpus']}")
    print(f"{'problem':24} {'rows':>5} {'#1':>7} {'#3':>7} {'#10':>7} {'#25':>7} {'#50':>7} {'median':>7}  you")
    for p in P:
        rows = snap["boards"][p]["rows"]
        sp = [r["speedup"] for r in rows if r.get("speedup")]
        def at(i):
            return f"{sp[i]:.3f}" if len(sp) > i else "-"
        me = next(((i + 1, r["speedup"]) for i, r in enumerate(rows) if r["team"].lower() == a.team.lower()), None)
        you = f"#{me[0]} x{me[1]:.3f}" if me else "-"
        print(f"{p:24} {len(rows):>5} {at(0):>7} {at(2):>7} {at(9):>7} {at(24):>7} {at(49):>7} {at(len(sp)//2):>7}  {you}")
    ov = snap["overall"]["rows"]
    me = next((r for r in ov if r["team"].lower() == a.team.lower()), None)
    print(f"overall: {len(ov)} teams; you: {me if me else 'not ranked'}")


if __name__ == "__main__":
    main()

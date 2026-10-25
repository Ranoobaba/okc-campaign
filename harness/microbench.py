"""Fast variant exploration: time several source variants of one submission on one or two
shapes in a single Modal container (graph replay, cold L2), next to the baseline.

  modal run harness/microbench.py --problem muse_attention_decode --base problems/muse_attention_decode/submission.py \
      --variants harness/variants_attn.json --cases "verify 1,decode 1"
variants json: {"name": [["old", "new"], ...], ...}; "" (empty list) = unchanged source.
"""

import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ROOT / "problems"
app = modal.App("okc-microbench")
image = (modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open", add_python=None)
         .env({"PYTHONUNBUFFERED": "1"}).add_local_dir(str(PROBLEMS), remote_path="/root/problems"))


def _import(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@app.function(image=image, gpu="H200", timeout=3600)
def run(problem, variants, cases, reps, check):
    import os, statistics, sys, traceback
    import torch
    pdir = f"/root/problems/{problem}"
    sys.path.insert(0, pdir)
    ref = _import(f"{pdir}/reference.py", "reference")
    base = _import(f"{pdir}/baseline.py", "baseline")
    dev = torch.device("cuda", 0)
    has_w = hasattr(ref, "generate_weights")
    weights = ref.generate_weights(ref.WEIGHTS_SEED) if has_w else None
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def clone(d):
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}

    def timeit(fn, data):
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                out = fn(data)
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn(data)
        torch.cuda.synchronize()
        ts = []
        for i in range(reps):
            flush.fill_(float(i)); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000)
        return statistics.median(ts), out

    def gen(case):
        kw = {k: v for k, v in case.items() if k not in ("load", "weight", "repeats", "router_load", "router_distinct")}
        if hasattr(ref, "case_fields"):
            kw.update(ref.case_fields(kw, weights))
        return ref.generate_input(**kw)

    def check_out(data, out):
        if hasattr(ref, "collect"):
            raw = ref.collect(data, out)
            return ref.check_implementation(data, raw, weights) if has_w else ref.check_implementation(data, raw)
        return ref.check_implementation(data, out, weights)

    results = {}
    base_state = base.prepare({k: v.clone() for k, v in weights.items()}) if has_w else None
    mods = {}
    for name, src in variants.items():
        d = f"/root/var_{name}"; os.makedirs(d, exist_ok=True)
        open(f"{d}/submission.py", "w").write(src)
        try:
            m = _import(f"{d}/submission.py", f"sub_{name}")
            st = m.prepare({k: v.clone() for k, v in weights.items()}) if (has_w and hasattr(m, "prepare")) else None
            mods[name] = (m, st)
        except Exception:
            results[name] = {"error": traceback.format_exc()[-1500:]}
            print(f"[{name}] import/prepare failed:\n{results[name]['error']}", flush=True)
    for case in cases:
        cname = case.get("load", "") + " | " + ", ".join(f"{k} {v}" for k, v in case.items() if k not in ("load", "weight", "seed", "repeats"))
        data = gen(case)
        tb, _ = timeit(lambda d: base.kernel_fn(d, base_state) if has_w else base.kernel_fn(d), clone(data))
        print(f"\n=== {cname}: baseline {tb:.1f} us", flush=True)
        for name, (m, st) in mods.items():
            try:
                d = clone(data)
                fn = (lambda dd: m.kernel_fn(dd, st)) if (has_w or len(__import__('inspect').signature(m.kernel_fn).parameters) > 1) else m.kernel_fn
                t, out = timeit(fn, d)
                msg = ""
                if check:
                    ok, msg = check_out(d, out)
                    msg = ("OK " if ok else "FAIL ") + msg[:120]
                print(f"  {name:28s} {t:8.1f} us  x{tb / t:.3f}  {msg}", flush=True)
                results.setdefault(name, {})[cname] = {"us": t, "speedup": tb / t, "check": msg}
            except Exception:
                err = traceback.format_exc()[-1200:]
                print(f"  {name:28s} ERROR {err.splitlines()[-1][:150]}", flush=True)
                results.setdefault(name, {})[cname] = {"error": err}
    return results


@app.local_entrypoint()
def main(problem: str, base: str, variants: str, cases: str = "", reps: int = 10, check: bool = True):
    src = Path(base).read_text()
    spec = json.loads(Path(variants).read_text())
    var_src = {}
    for name, edits in spec.items():
        s = src
        for old, new in edits:
            assert old in s, f"variant {name}: pattern not found: {old[:60]}"
            s = s.replace(old, new)
        var_src[name] = s
    benches = json.loads((PROBLEMS / problem / "meta.json").read_text())["benchmarks"]
    want = [c.strip() for c in cases.split(",") if c.strip()]
    sel = [b for b in benches if not want or any(w in b.get("load", "") or w in json.dumps(b) for w in want)]
    res = run.remote(problem, var_src, sel, reps, check)
    out = ROOT / "results" / problem / f"micro_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("saved", out)

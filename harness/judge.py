"""Local replica of the Open Kernel Challenge judge, run on a Modal H200 inside the official
`lmsysorg/sglang:dev-muse-spark-open` image.

What it reproduces from TASK.md "How a run is timed":
  * each shape captured as a CUDA graph (baseline and submission), replayed on fresh inputs
  * a cold L2 before every timed replay (a 256 MB buffer is written first)
  * paired ratios: reps taken two at a time with the order swapped; the shape speedup is the
    median of pair ratios; a 95% interval from a bootstrap over pairs
  * the graph replay fixed cost (an empty graph) subtracted from both sides
  * every timed call checked against the reference (configurable: --check-reps)
  * loads: load time = sum(weight * shape time); problem speedup = geomean over loads

Usage (from the campaign root):
  modal run harness/judge.py --problem muse_bf16_proj --submission problems/muse_bf16_proj/submission.py
  options: --reps 20 --check-reps 3 --only "decode 1" --skip-tests --profile --tag name --no-bench
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ROOT / "problems"

app = modal.App("okc-judge")
image = (
    modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open", add_python=None)
    .env({"PYTHONUNBUFFERED": "1", "TORCH_CUDA_ARCH_LIST": "9.0"})
    .add_local_dir(str(PROBLEMS), remote_path="/root/problems")
)

HW = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
FLUSH_BYTES = 256 << 20


def _import(path: str, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case_kwargs(case: dict) -> dict:
    return {k: v for k, v in case.items() if k not in ("load", "weight", "repeats", "router_load", "router_distinct")}


def _copy_into(dst, src):
    """Copy fresh inputs into the static buffers a graph was captured on."""
    import torch

    for k, v in src.items():
        if isinstance(v, torch.Tensor):
            dst[k].copy_(v)
        elif isinstance(v, dict):
            pass  # config: identical across seeds of a case


def _clone_inputs(data):
    import torch

    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}


@app.function(image=image, gpu="H200", timeout=3 * 3600, memory=65536)
def run(problem: str, sub_src: str, tests: list, benches: list, opts: dict) -> dict:
    import inspect
    import math
    import statistics
    import sys
    import traceback

    import torch

    t_start = time.time()
    pdir = f"/root/problems/{problem}"
    sys.path.insert(0, pdir)
    subdir = "/root/sub"
    os.makedirs(subdir, exist_ok=True)
    with open(f"{subdir}/submission.py", "w") as f:
        f.write(sub_src)
    ref = _import(f"{pdir}/reference.py", "reference")
    base = _import(f"{pdir}/baseline.py", "baseline")
    sub = _import(f"{subdir}/submission.py", "submission")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    report: dict = {"problem": problem, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                    "tests": [], "shapes": [], "ok": True}

    has_weights = hasattr(ref, "generate_weights")
    weights = ref.generate_weights(ref.WEIGHTS_SEED) if has_weights else None

    def prep(mod, w):
        if w is None or not hasattr(mod, "prepare"):
            return None
        return mod.prepare({k: v.clone() for k, v in w.items()})

    t0 = time.time()
    base_state = prep(base, weights)
    torch.cuda.synchronize()
    t_base_prep = time.time() - t0
    t0 = time.time()
    try:
        sub_state = prep(sub, weights)
        torch.cuda.synchronize()
    except Exception:
        report["ok"] = False
        report["error"] = "prepare failed:\n" + traceback.format_exc()
        return report
    t_sub_prep = time.time() - t0
    report["prepare_s"] = {"baseline": round(t_base_prep, 1), "submission": round(t_sub_prep, 1)}
    if t_sub_prep > 300:
        report["ok"] = False
        report["error"] = f"prepare took {t_sub_prep:.0f}s > 300s"
        return report

    n_params = len(inspect.signature(sub.kernel_fn).parameters)

    def call(mod, state, data):
        if n_params >= 2 or mod is base and has_weights:
            return mod.kernel_fn(data, state)
        return mod.kernel_fn(data)

    def call_sub(data):
        return sub.kernel_fn(data, sub_state) if n_params >= 2 else sub.kernel_fn(data)

    def call_base(data):
        return base.kernel_fn(data, base_state) if has_weights else base.kernel_fn(data)

    def gen(case: dict, seed: int):
        kw = dict(_case_kwargs(case))
        kw["seed"] = seed
        if hasattr(ref, "case_fields"):
            kw.update(ref.case_fields(kw, weights))
        return ref.generate_input(**kw)

    def check(data, out):
        if hasattr(ref, "collect"):
            raw = ref.collect(data, out)
            return ref.check_implementation(data, raw, weights) if has_weights else ref.check_implementation(data, raw)
        return ref.check_implementation(data, out, weights)

    # ---------------------------------------------------------------- tests
    if not opts.get("skip_tests"):
        for i, case in enumerate(tests):
            name = ", ".join(f"{k} {v}" for k, v in case.items() if k != "seed")
            t0 = time.time()
            try:
                data = gen(case, case.get("seed", 9000 + i))
                d = _clone_inputs(data)  # the kernel writes the cache in place: check that same dict
                out = call_sub(d)
                torch.cuda.synchronize()
                ok, msg = check(d, out)
            except Exception:
                ok, msg = False, "exception:\n" + traceback.format_exc()
            report["tests"].append({"case": name, "ok": bool(ok), "msg": msg, "s": round(time.time() - t0, 1)})
            print(f"[test] {'PASS' if ok else 'FAIL'} {name}: {msg if len(msg) < 400 else msg[:400]}", flush=True)
            if not ok:
                report["ok"] = False
                if opts.get("stop_on_fail", True):
                    report["error"] = f"test failed: {name}: {msg}"
                    return report

    if opts.get("no_bench"):
        return report

    # ---------------------------------------------------------------- timing
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)

    def flush_l2(i):
        flush.fill_(float(i))

    # Graph replay fixed cost.
    empty = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(empty):
            pass
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fixed = []
    for i in range(20):
        flush_l2(i)
        e0.record()
        empty.replay()
        e1.record()
        torch.cuda.synchronize()
        fixed.append(e0.elapsed_time(e1) * 1000)
    fixed_us = statistics.median(fixed)
    report["graph_fixed_us"] = round(fixed_us, 2)

    def capture(fn, static):
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                out = fn(static)
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn(static)
        torch.cuda.synchronize()
        return g, out

    reps = int(opts.get("reps", 20))
    reps += reps % 2
    check_reps = int(opts.get("check_reps", 3))
    only = opts.get("only")
    loads: dict[str, dict] = {}
    for si, case in enumerate(benches):
        load = case.get("load", f"case {si}")
        if only and only not in load and only not in json.dumps(case):
            continue
        name = ", ".join(f"{k} {v}" for k, v in _case_kwargs(case).items() if k != "seed")
        weight = float(case.get("weight", 1))
        seed0 = int(case.get("seed", 100 + si))
        t0 = time.time()
        try:
            data = gen(case, seed0)
            static_b, static_s = _clone_inputs(data), _clone_inputs(data)
            g_b, out_b = capture(call_base, static_b)
            g_s, out_s = capture(call_sub, static_s)
        except Exception:
            report["ok"] = False
            report["error"] = f"capture failed on {name}:\n" + traceback.format_exc()
            print(report["error"], flush=True)
            return report
        tb, ts = [], []
        fails = []
        for r in range(reps):
            fresh = gen(case, seed0 + 1000 * (r + 1)) if r > 0 else data
            _copy_into(static_b, fresh)
            _copy_into(static_s, fresh)
            order = ((g_b, tb), (g_s, ts)) if r % 2 == 0 else ((g_s, ts), (g_b, tb))
            for g, acc in order:
                flush_l2(r)
                e0.record()
                g.replay()
                e1.record()
                torch.cuda.synchronize()
                acc.append(max(e0.elapsed_time(e1) * 1000 - fixed_us, 0.01))
            if r < check_reps:
                try:
                    ok, msg = check(static_s, out_s)  # static_s holds the fresh inputs plus the kernel's cache writes
                except Exception:
                    ok, msg = False, "exception:\n" + traceback.format_exc()
                if not ok:
                    fails.append(msg)
                    report["ok"] = False
                    print(f"[bench] FAIL check rep {r} on {name}: {msg[:300]}", flush=True)
        pairs = [(tb[2 * j] + tb[2 * j + 1]) / (ts[2 * j] + ts[2 * j + 1]) for j in range(reps // 2)]
        speed = statistics.median(pairs)
        # Bootstrap 95% interval over pairs.
        import random

        rng = random.Random(0)
        boots = sorted(statistics.median(rng.choices(pairs, k=len(pairs))) for _ in range(400))
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
        base_us = statistics.median(tb)
        sub_us = base_us / speed
        floor = None
        if hasattr(ref, "speed_of_light_ns"):
            try:
                floor = ref.speed_of_light_ns(hw=HW, **_case_kwargs(case)) / 1000
            except Exception:
                floor = None
        row = {"load": load, "case": name, "weight": weight, "base_us": round(base_us, 2), "sub_us": round(sub_us, 2),
               "speedup": round(speed, 4), "ci": [round(lo, 4), round(hi, 4)], "floor_us": round(floor, 2) if floor else None,
               "sub_min_us": round(min(ts), 2), "base_min_us": round(min(tb), 2), "fails": fails[:2], "s": round(time.time() - t0, 1)}
        report["shapes"].append(row)
        print(f"[bench] {load:>10} | {name:<70} | base {base_us:8.1f} us | sub {sub_us:8.1f} us | "
              f"x{speed:.3f} [{lo:.3f},{hi:.3f}] | floor {floor if floor is None else round(floor,1)}", flush=True)
        L = loads.setdefault(load, {"base": 0.0, "sub": 0.0})
        L["base"] += weight * base_us
        L["sub"] += weight * sub_us
        del g_b, g_s, static_b, static_s, out_b, out_s
        torch.cuda.empty_cache()

    if loads:
        ratios = [L["base"] / L["sub"] for L in loads.values()]
        geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        report["loads"] = {k: {"base_us": round(v["base"], 1), "sub_us": round(v["sub"], 1), "speedup": round(v["base"] / v["sub"], 4)}
                           for k, v in loads.items()}
        report["speedup"] = round(geo, 4)
        print(f"\n=== {problem}: geomean speedup over loads = x{geo:.4f}  (ok={report['ok']}) ===", flush=True)
        for k, v in report["loads"].items():
            print(f"  {k:>10}: base {v['base_us']:10.1f} us  sub {v['sub_us']:10.1f} us  x{v['speedup']:.4f}")

    # ---------------------------------------------------------------- profile
    if opts.get("profile"):
        from torch.profiler import ProfilerActivity, profile

        prof_out = {}
        for si, case in enumerate(benches):
            load = case.get("load", f"case {si}")
            if only and only not in load and only not in json.dumps(case):
                continue
            name = ", ".join(f"{k} {v}" for k, v in _case_kwargs(case).items() if k != "seed")
            data = gen(case, int(case.get("seed", 100 + si)))
            for label, fn in (("submission", call_sub), ("baseline", call_base)):
                d = _clone_inputs(data)
                for _ in range(3):
                    fn(d)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as p:
                    for _ in range(5):
                        fn(d)
                    torch.cuda.synchronize()
                table = p.key_averages().table(sort_by="cuda_time_total", row_limit=18)
                prof_out[f"{name} | {label}"] = table
                print(f"\n--- profile: {name} | {label} ---\n{table}", flush=True)
        report["profile"] = prof_out

    report["total_s"] = round(time.time() - t_start, 1)
    return report


@app.local_entrypoint()
def main(problem: str, submission: str, reps: int = 20, check_reps: int = 3, only: str = "", skip_tests: bool = False,
         profile: bool = False, tag: str = "", no_bench: bool = False, stop_on_fail: bool = True):
    pdir = PROBLEMS / problem
    tests = json.loads((pdir / "tests.json").read_text())
    benches = json.loads((pdir / "meta.json").read_text())["benchmarks"]
    src = Path(submission).read_text()
    opts = {"reps": reps, "check_reps": check_reps, "only": only or None, "skip_tests": skip_tests,
            "profile": profile, "no_bench": no_bench, "stop_on_fail": stop_on_fail}
    rep = run.remote(problem, src, tests, benches, opts)
    out_dir = ROOT / "results" / problem
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = tag or Path(submission).stem
    path = out_dir / f"{stamp}_{name}.json"
    rep["submission"] = str(submission)
    rep["tag"] = name
    path.write_text(json.dumps(rep, indent=1))
    if rep.get("error"):
        print("\nERROR:", rep["error"][:3000])
    print(f"\nsaved {path}")
    print(f"RESULT {problem} {name} ok={rep.get('ok')} speedup={rep.get('speedup')}")

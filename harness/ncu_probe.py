"""Run one eager call of a submission under Nsight Compute inside the judge image and print the
key sections for the kernels matching a regex.

  modal run harness/ncu_probe.py --problem muse_moe_w4a16 --submission F --case-index 0 --kernel-regex w4a16
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ROOT / "problems"
app = modal.App("okc-ncu")
image = (modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open", add_python=None)
         .env({"PYTHONUNBUFFERED": "1"}).add_local_dir(str(PROBLEMS), remote_path="/root/problems"))

DRIVER = r'''
import sys, json, importlib.util, torch
problem, sub_path, case_json = sys.argv[1], sys.argv[2], sys.argv[3]
pdir = f"/root/problems/{problem}"; sys.path.insert(0, pdir)
def imp(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ref = imp(f"{pdir}/reference.py", "reference"); sub = imp(sub_path, "submission")
case = json.loads(case_json)
kw = {k: v for k, v in case.items() if k not in ("load", "weight", "repeats", "router_load", "router_distinct")}
has_w = hasattr(ref, "generate_weights")
weights = ref.generate_weights(ref.WEIGHTS_SEED) if has_w else None
if hasattr(ref, "case_fields"): kw.update(ref.case_fields(kw, weights))
data = ref.generate_input(**kw)
st = sub.prepare(weights) if (has_w and hasattr(sub, "prepare")) else None
import inspect
call = (lambda d: sub.kernel_fn(d, st)) if len(inspect.signature(sub.kernel_fn).parameters) > 1 else sub.kernel_fn
for _ in range(2): call(data)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
call(data); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("DRIVER_DONE")
'''


@app.function(image=image, gpu="H200", timeout=1800)
def run(problem, sub_src, case, kernel_regex, sections):
    import os, subprocess
    os.makedirs("/root/sub", exist_ok=True)
    open("/root/sub/submission.py", "w").write(sub_src)
    open("/root/sub/driver.py", "w").write(DRIVER)
    cmd = ["ncu", "--clock-control", "none", "--profile-from-start", "off", "--kernel-name", f"regex:{kernel_regex}", "--launch-count", "4"]
    for s in sections:
        cmd += ["--section", s]
    cmd += ["python", "/root/sub/driver.py", problem, "/root/sub/submission.py", json.dumps(case)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    return r.stdout[-30000:] + "\nSTDERR:\n" + r.stderr[-4000:]


@app.local_entrypoint()
def main(problem: str, submission: str, case_index: int = 0, kernel_regex: str = ".", sections: str = "SpeedOfLight,Occupancy,WarpStateStats,LaunchStats,SchedulerStats,MemoryWorkloadAnalysis"):
    benches = json.loads((PROBLEMS / problem / "meta.json").read_text())["benchmarks"]
    out = run.remote(problem, Path(submission).read_text(), benches[case_index], kernel_regex, sections.split(","))
    print(out)

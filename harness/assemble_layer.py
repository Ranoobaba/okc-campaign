"""Build problems/muse_decode_layer/submission.py (and the prefill variant) from the three
component kernels plus the layer glue in harness/layer_glue.py. One file per submission is the
judge's format, so components are concatenated with their entry points renamed."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "problems"


def component(path, prefix):
    src = Path(path).read_text()
    src = re.sub(r"^# okc: problem=.*\n", "", src)
    src = re.sub(r'^""".*?"""\n', "", src, count=1, flags=re.S)  # module docstring
    src = src.replace("def prepare(", f"def {prefix}_prepare(").replace("def kernel_fn(", f"def {prefix}_kernel_fn(")
    # Drop the TypedDict declarations (they only document the component's own I/O).
    src = re.sub(r"\nclass (Weights|Config|Inputs)\(TypedDict\):\n(?:    .*\n|\n)*", "\n", src)
    return f"\n# ====================== component: {prefix} ======================\n" + src


def build(problem):
    out = [f"# okc: problem={problem}\n",
           '"""Whole decoder layer: our projections, attention and MoE kernels with SGLang\'s fused\n'
           'norm/residual/routing modules in between (assembled by harness/assemble_layer.py)."""\n',
           "from typing import TypedDict\nimport torch\nimport torch.nn.functional as F\nimport triton\nimport triton.language as tl\n"]
    out.append(component(P / "muse_bf16_proj/submission.py", "proj"))
    out.append(component(P / "muse_attention_decode/submission.py", "attn"))
    out.append(component(P / "muse_moe_w4a16/submission.py", "moe"))
    out.append("\n# ====================== layer glue ======================\n")
    out.append((ROOT / "harness/layer_glue.py").read_text())
    text = "".join(out)
    # Imports from components must all sit at top; triton/torch imports repeated are harmless.
    (P / problem / "submission.py").write_text(text)
    print(f"wrote {P / problem / 'submission.py'} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    for prob in (sys.argv[1:] or ["muse_decode_layer"]):
        build(prob)

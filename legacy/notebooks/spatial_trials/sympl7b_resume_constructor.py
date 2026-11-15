# Run as one Python cell in the current Kaggle session.
from pathlib import Path
import subprocess

path = SCRIPTS / "sympl7b_device_map.py"
old = """    with init_empty_weights():
        empty = Qwen2_5_VLForConditionalGeneration.from_config(
            config, torch_dtype=torch.float16, attn_implementation='sdpa')"""
new = """    config._attn_implementation = 'sdpa'
    with init_empty_weights():
        empty = Qwen2_5_VLForConditionalGeneration(config)"""
source = path.read_text(encoding="utf-8-sig")
if old in source:
    source = source.replace(old, new, 1)
    compile(source, str(path), "exec")
    path.write_text(source, encoding="utf-8")
elif new not in source:
    raise RuntimeError("File differs from the notebook inspected; no changes made.")
print("Patched model construction. Resuming SymPL...")
subprocess.run(
    [PY, "-u", str(SCRIPTS / "worker.py"), str(CONFIG), "0", "1"],
    cwd=REPO, env=ENV, check=True,
)

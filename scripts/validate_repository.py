"""Check profiles, configs and thin notebook syntax without GPU dependencies."""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sc203_kat.runtime.config import load_config
from sc203_kat.methods import describe

def main():
    sources = [p for name in ("src","profiles","scripts") for p in (ROOT/name).rglob("*.py")]
    for path in sources: compile(path.read_text(encoding="utf-8-sig"),str(path),"exec")
    notebooks = sorted((ROOT/"notebooks").rglob("*.ipynb"))
    assert len(notebooks)==13
    expected_ablation={'sanity.ipynb','01_sam3_oneref.ipynb','02_orientanything_v2.ipynb'}
    assert {p.name for p in (ROOT/'notebooks/apc_ablation').glob('*.ipynb')}==expected_ablation
    for path in notebooks:
        assert path.stat().st_size<25000, f"Notebook should be thin: {path}"
        text=path.read_text(encoding="utf-8")
        assert "KAT_INPUT" not in text and "zipfile" not in text, f"Obsolete code loading in {path}"
        assert "https://github.com/leetranwyng/sc203-kat.git" in text
        for i,cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
            if cell["cell_type"]=="code":
                compile("".join(cell["source"]),f"{path}:cell{i}","exec")
                assert not cell.get("outputs")
    configs=list((ROOT/"configs/experiments").glob("*.json"))
    assert len(configs)==10
    for path in configs:
        config=load_config(path,ROOT); plan=describe(config["method"],config,root=ROOT)
        for stage in plan["stages"]: assert (Path(plan["profile_dir"])/stage).is_file()
    for path in (ROOT/'configs/ablations/experiments').glob('*.json'):
        conf=json.loads(path.read_text()); assert 'baseline' not in conf['variants']
    for folder in ("src","profiles","configs","notebooks"):
        for path in (ROOT/folder).rglob("*"):
            if path.is_file() and path.suffix in {".py",".json",".ipynb"}:
                assert not re.search(r"(?:hf_[A-Za-z0-9]{25,}|ghp_[A-Za-z0-9]{30,})",path.read_text(encoding="utf-8-sig")),f"Potential credential in {path}"
    print(f"PASS: {len(sources)} Python files, thirteen thin notebooks, ten baseline configs plus three ablation configs, no credential literals")
if __name__=="__main__": main()

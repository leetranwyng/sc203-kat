"""Local repository initialization helper; never creates GitHub forks/remotes."""
import json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def git(*args, capture=False):
    if capture: return subprocess.check_output(["git","-C",str(ROOT),*args],text=True).strip()
    subprocess.run(["git","-C",str(ROOT),*args],check=True)
def main():
    if (ROOT/".git").exists():
        raise SystemExit("Git already initialized; no changes made")
    git("init","-b","main")
    upstreams=json.loads((ROOT/"manifests/upstreams.json").read_text(encoding="utf8"))
    for item in upstreams["repositories"]:
        git("update-index","--add","--cacheinfo","160000",item["pinned_commit"],item["path"])
    git("add",".")
    git("submodule","absorbgitdirs")
    for key,value in [("user.name","SC203 KAT"),("user.email","sc203-kat@localhost")]:
        result=subprocess.run(["git","-C",str(ROOT),"config","--get",key],capture_output=True,text=True)
        if result.returncode: git("config",key,value)
    git("commit","-m","Initialize SC203-KAT experiment framework and migrated baselines")
    git("tag","v0.1.0")
    print("Created local main branch and v0.1.0; no remote was created")
if __name__=="__main__": main()

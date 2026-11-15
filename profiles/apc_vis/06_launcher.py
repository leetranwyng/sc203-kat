from apc_launcher import launch as launch_full
def launch():
    return launch_full(CFG,PY,SCRIPTS,ENV,retry_failed=CFG.get("retry_failed",False))

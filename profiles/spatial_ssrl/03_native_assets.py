(SCRIPTS/'ssrl_assets.py').write_text((PROFILE_DIR/"sources/ssrl_assets.py").read_text(encoding="utf8"),encoding="utf8")
subprocess.run([PY,str(SCRIPTS/'ssrl_assets.py'),str(CONFIG)],env=ENV,check=True)
CFG=json.loads(CONFIG.read_text())
CONFIG.write_text(json.dumps(CFG))

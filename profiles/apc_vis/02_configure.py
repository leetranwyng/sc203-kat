CFG=json.loads((PROFILE_DIR/'method_defaults.json').read_text(encoding='utf8'))
CFG.update(METHOD_OPTIONS)
CFG.update(work=str(WORK),repo=str(REPO),output=str(OUTPUT_DIR),asset_input=MODEL_INPUT,dataset_input=DATASET_INPUT,resume_input=RESUME_INPUT,sanity=SANITY,run_kind='sanity' if SANITY else 'full')
if SANITY:
    CFG.update(limit=2,expected_rows=6,expected_circular_groups=3,expected_task_counts={'left_right':2,'visibility':2,'facing':2},run_timeout_seconds=1800,trace_keep_per_task=2,trace_max_total_mib=512)
OUT=Path(OUTPUT_DIR);OUT.mkdir(parents=True,exist_ok=True)
CONFIG=WORK/'config.json'
with setup_stage('resume_configuration',OUT,CFG):prepare_resume(CFG,WORK)
CONFIG.write_text(json.dumps(CFG,indent=2),encoding='utf8')
(OUT/'config.json').write_text(json.dumps(CFG,indent=2),encoding='utf8')

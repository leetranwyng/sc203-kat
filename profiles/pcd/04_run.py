driver=(PROFILE_DIR/"sources/driver.py").read_text(encoding="utf8")
WORK_DIR=str(WORK)
values={k:globals()[k] for k in ['WORK_DIR','MODE','MODEL_ID','MODEL_REVISION','DATASET_ID','DATASET_REVISION','TSV_FILENAME','ALPHA','NUM_WORKERS','SEED','COMPUTE_BASELINE','MIN_PIXELS','MAX_PIXELS','MODEL_DIR','TSV_PATH','SANITY_QID','OUTPUT_DIR','RESUME_FROM']}
prefix='from pathlib import Path\n'+''.join(k+' = '+('Path('+repr(str(v))+')' if isinstance(v,Path) else repr(v))+'\n' for k,v in values.items())
job=WORK/'run_pcd.py'
job.write_text(prefix+driver,encoding='utf8')
subprocess.run([PY,'-u',str(job)],env=ENV,check=True)

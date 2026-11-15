with setup_stage('trace_reference_patch_and_data',OUT,CFG):
    run_logged('trace_patch',[PY,str(SCRIPTS/'apc_trace_patches.py'),str(REPO),str(WORK/'trace_patches.json')],OUT,env=ENV)
    run_logged('reference_patch',[PY,str(SCRIPTS/'apc_full_reference.py'),str(REPO),str(WORK/'reference_patches.json')],OUT,env=ENV)
    DATA_ENV=ENV.copy();DATA_ENV.pop('HF_HUB_OFFLINE',None);DATA_ENV.pop('TRANSFORMERS_OFFLINE',None)
    run_logged('prepare_data',[PY,str(SCRIPTS/'apc_prepare_data.py'),str(CONFIG)],OUT,env=DATA_ENV,cwd=REPO)
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['selected_rows']==CFG['expected_rows'],manifest['selected_rows']
    assert manifest['circular_groups']==CFG['expected_circular_groups']
    assert manifest['tasks']==CFG['expected_task_counts']
    print('Selected:',manifest['tasks'])

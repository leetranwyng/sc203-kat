CONFIG.write_text(json.dumps(CFG))
subprocess.run([PY,str(SCRIPTS/'data.py'),str(CONFIG)],cwd=REPO,env=ENV,check=True)
ENV.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
(OUT/'environment.txt').write_text(subprocess.check_output([str(UV),'pip','freeze','--python',PY],env=ENV,text=True))
shutil.copy2(WORK/'gpu_check.txt',OUT/'gpu_check.txt')
workers=1  # One model spans both GPUs.
processes=[]
try:
    for rank in range(workers):
        env=ENV.copy()
        env['CUDA_VISIBLE_DEVICES']=','.join(GPU_IDS)
        log=(OUT/f'worker_{rank}.log').open('a')
        child=subprocess.Popen([PY,'-u',str(SCRIPTS/'worker.py'),str(CONFIG),str(rank),str(workers)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=env)
        processes.append((child,log))
    while any(p.poll() is None for p,_ in processes):
        for i,(p,_) in enumerate(processes):
            if p.poll() not in (0,None):raise RuntimeError((OUT/f'worker_{i}.log').read_bytes()[-4000:].decode(errors='replace'))
        print('Saved',sum(1 for _ in OUT.glob('traces/*/result.json')),'/',json.loads((OUT/'manifest.json').read_text())['selected_rows'],flush=True)
        time.sleep(15)
    for i,(p,_) in enumerate(processes):
        if p.returncode:raise RuntimeError((OUT/f'worker_{i}.log').read_bytes()[-4000:].decode(errors='replace'))
finally:
    for p,log in processes:
        if p.poll() is None:p.terminate()
        p.wait();log.close()

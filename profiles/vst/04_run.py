from IPython.display import FileLink, display

# Streaming console is bounded; full per-row generations go to predictions.jsonl.
with (OUT/'worker.log').open('a', encoding='utf8', buffering=1) as log:
    process = subprocess.Popen([str(PY),'-u',str(WORK/'vst_runner.py'),str(CONFIG)],
                               env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               encoding='utf8',errors='replace',bufsize=1)
    try:
        for line in process.stdout:
            log.write(line)
            print(line,end='',flush=True)
    except KeyboardInterrupt:
        import signal
        process.send_signal(signal.SIGINT)
        process.wait()
    rc = process.wait()

archive = OUT/'checkpoint.zip'
if archive.is_file():
    display(FileLink(os.path.relpath(archive, '/kaggle/working')))
if rc:
    print(f'Worker exited with code {rc}. Inspect worker.log and last_error.json.')
    if not archive.is_file():
        raise RuntimeError('Run failed before the first checkpoint. See worker.log.')
    raise RuntimeError(f'VST worker exited with code {rc}; checkpoint preserved at {archive}. Inspect {OUT / "worker.log"}.')
else:
    print('See summary.json and its complete/attempted_rows fields for coverage.')

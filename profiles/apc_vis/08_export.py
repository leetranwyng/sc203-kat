from apc_launcher import collect_reports
collect_reports(WORK,OUT)
try:
    packages=subprocess.check_output([str(UV),'pip','freeze','--python',PY],env=ENV,text=True)
    (OUT/'environment.txt').write_text(packages,encoding='utf-8')
except Exception as exc:
    print('Environment report:',repr(exc))
export_full(OUT,CFG)
show_downloads(OUT)
print('Download results and traces from this run. Check coverage before comparing accuracy.')

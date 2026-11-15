REPORT=launch()
print('Status:',REPORT['completion_status'],'technical:',REPORT['technical_status'])
print('Attempted:',REPORT['attempted_rows'],'/',REPORT['expected_rows'],'=',round(REPORT['attempted_percent'],2),'%')
print('Valid:',REPORT['valid_predictions'],'errors:',REPORT['pipeline_errors'],'invalid:',REPORT['invalid_outputs'])
print('Row accuracy:',REPORT['row_accuracy'],'Circular accuracy:',REPORT['circular_accuracy'])
print('Per task:',json.dumps(REPORT['by_task'],indent=2))
if REPORT['completion_status']=='PAUSED':
    print('Reached the worker time budget. Results so far are in Output.')
elif not REPORT['complete']:
    print('Inspect errors/coverage in full_report.json; accuracy keeps missing/error rows in denominator.')

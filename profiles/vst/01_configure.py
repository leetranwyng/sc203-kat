MODE='sanity' if SANITY else 'full'
RUN_NAME=METHOD_OPTIONS.get('run_name','vst7b_'+MODE)
PROTOCOL=METHOD_OPTIONS.get('protocol','both')
THINKING=METHOD_OPTIONS.get('thinking',True)
MAX_NEW_TOKENS=METHOD_OPTIONS.get('max_new_tokens',1280)
MAX_IMAGE_TOKENS=METHOD_OPTIONS.get('max_image_tokens',1280)
SESSION_BUDGET_MINUTES=METHOD_OPTIONS.get('session_budget_minutes',420)
MAX_ROWS_THIS_SESSION=METHOD_OPTIONS.get('max_rows_this_session',0)

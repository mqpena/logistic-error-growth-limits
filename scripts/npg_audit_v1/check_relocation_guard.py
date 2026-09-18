from pathlib import Path
import json,sys
# The guard should reject the forbidden path before attempting the file access.
blocked=False
try:
    Path('/legacy/LogisticErrorGrowth-Program/this_path_need_not_exist').read_bytes()
except RuntimeError as e:
    blocked='Legacy project runtime access denied' in str(e)
if not blocked:
    raise RuntimeError('Archive guard failed to block the old project')
print(json.dumps({'legacy_access_blocked':True,'python':sys.version.split()[0]}))

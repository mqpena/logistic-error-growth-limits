#!/usr/bin/env python3
"""Run a local analysis while denying access to the relocatable legacy project.

This does not modify, rename, or hide the author's legacy archive. It makes
accidental runtime dependency on that archive fail visibly during the audit.
"""
from pathlib import Path
import os
import runpy
import sys

FORBIDDEN_FRAGMENT = 'LogisticErrorGrowth-Program'
sys.dont_write_bytecode = True

def forbidden(value):
    if isinstance(value, (str, bytes, os.PathLike)):
        return FORBIDDEN_FRAGMENT in os.fsdecode(value)
    if isinstance(value, (tuple, list)):
        return any(forbidden(x) for x in value)
    return False

def guard(event, args):
    # Restrict paths actually opened or dispatched, not provenance text read
    # from a locally copied manifest containing a historical source filename.
    if event in {'open', 'os.listdir', 'os.scandir', 'os.chdir', 'os.exec',
                 'os.posix_spawn', 'subprocess.Popen'} and forbidden(args):
        raise RuntimeError(f'Legacy project runtime access denied ({event})')

if len(sys.argv) < 2:
    raise SystemExit('usage: with_archive_guard.py SCRIPT [SCRIPT_ARGS ...]')
script = Path(sys.argv[1]).resolve()
if FORBIDDEN_FRAGMENT in str(script):
    raise SystemExit('Refusing legacy script path')
sys.argv = [str(script)] + sys.argv[2:]
sys.path.insert(0, str(script.parent))
sys.addaudithook(guard)
runpy.run_path(str(script), run_name='__main__')

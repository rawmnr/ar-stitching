import os, subprocess, shutil, json
env = os.environ.copy()
env['OPENCODE_PERMISSION'] = '{"*": "allow"}'
print(f"OPENCODE_PERMISSION={env['OPENCODE_PERMISSION']}")
subprocess.run([shutil.which('opencode'), 'run', 'read README.md'], env=env)

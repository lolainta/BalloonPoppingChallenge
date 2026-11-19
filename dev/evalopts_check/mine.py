"""List (or with 'kill', SIGTERM) the processes whose environment carries TAG=VALUE (default EVALOPTS_MINE=1):
forkserver / pool workers do not name the script on their command line.
.venv/bin/python dev/evalopts_check/mine.py [kill] [TAG=VALUE]"""
import os, signal, subprocess, sys

tag = next((a for a in sys.argv[1:] if "=" in a), "EVALOPTS_MINE=1").encode()
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
    except OSError:
        continue
    if tag in env and int(pid) != os.getpid():
        print(subprocess.run(["ps", "-o", "pid,etime,time,pcpu,ni,args", "--no-headers", "-p", pid], capture_output=True,
                             text=True).stdout.strip()[:160])
        if "kill" in sys.argv[1:]:
            os.kill(int(pid), signal.SIGTERM)

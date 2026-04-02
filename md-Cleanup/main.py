import os
import subprocess
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_script = os.path.join(script_dir, "2-Pipeline", "script-pipeline.py")
    cmd = [sys.executable, pipeline_script, *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()

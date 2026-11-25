"""Build and update the yolo services when the training queue is idle."""

import subprocess
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parent
    compose = ["docker", "compose", "--project-directory", str(project)]
    subprocess.run([*compose, "build"], check=True)
    services = subprocess.run([*compose, "ps", "--status", "running", "--services"],
                              check=True, capture_output=True, text=True).stdout.splitlines()
    if "backend" in services:
        check = """import json
import urllib.request

with urllib.request.urlopen('http://127.0.0.1:8000/api/runs', timeout=10) as response:
    jobs = json.load(response)
active = [record['id'] for record in jobs if record['status'] == 'running']
if active:
    print('Update delayed while training is active:', ', '.join(active))
    raise SystemExit(2)
"""
        result = subprocess.run([*compose, "exec", "-T", "backend", "python", "-"], input=check, text=True)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    subprocess.run([*compose, "up", "-d", "--no-build", "--wait"], check=True)


if __name__ == "__main__":
    main()

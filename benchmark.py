import os
import sys
import shutil
import subprocess
from pathlib import Path

# ContrastDTA benchmark launcher
# - Runs the repo's entrypoint in the ContrastDTA conda environment
# - Streams logs unbuffered
# - Pass through any extra CLI args you provide to this wrapper

MODEL_NAME = "ContrastDTA"
ENV_NAME = "ContrastDTA2"
# Adjust this if your entrypoint differs
SCRIPT_PATH = "/home/patrick/Desktop/ContrastDTA2/main.py"
DEFAULT_ARGS = ["--config", "/home/patrick/Desktop/ContrastDTA2/config.yaml"]  # default config; override via CLI if needed


def main():
    extra_args = sys.argv[1:]

    script = Path(SCRIPT_PATH)
    if not script.is_file():
        print(f"ERROR: Entry script not found: {script}")
        print("Please update SCRIPT_PATH in ContrastDTA_benchmark.py to the correct file.")
        sys.exit(1)

    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"

    cwd = str(script.parent)

    # Run with the currently active Python environment; no conda run
    python_exe = sys.executable or "python"
    cmd = [python_exe, "-u", str(script), *DEFAULT_ARGS, *extra_args]

    print(f"Running {MODEL_NAME}...\nCommand: {' '.join(cmd)}\nCWD: {cwd}")
    proc = subprocess.run(cmd, cwd=cwd, env=base_env, stdout=sys.stdout, stderr=sys.stderr)
    if proc.returncode != 0:
        print(f"{MODEL_NAME} exited with code {proc.returncode}")
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()

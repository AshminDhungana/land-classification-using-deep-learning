import glob
import subprocess
import sys

"""Run all scripts in the scripts directory in order."""

def run_all_scripts():
    scripts = sorted(glob.glob("./scripts/*.py"))

    for script in scripts:
        print(f"Running {script}...")
        subprocess.run([sys.executable, script])
        print(f"Finished {script}\n")


if __name__ == "__main__":
    run_all_scripts()

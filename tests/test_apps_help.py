"""Run `--help` on the built product binaries and check the exit codes.

The B200 build is a separate compile target; this test guards that the built
binaries start on their own and exit cleanly for `--help` without touching
unavailable devices. It runs for any target that produced the apps.
"""

import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: test_apps_help.py NINFER NINFER_SERVE NINFER_PERPLEXITY")
        return 2
    for binary in sys.argv[1:]:
        result = subprocess.run(
            [binary, "--help"], capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"FAIL: {binary} --help exited {result.returncode}")
            print(result.stderr[-2000:])
            return 1
        if not result.stdout.strip():
            print(f"FAIL: {binary} --help produced no output")
            return 1
        for marker in ("no kernel image", "invalid device function"):
            if marker in result.stderr:
                print(f"FAIL: {binary} --help reported '{marker}'")
                return 1
        print(f"ok: {binary} --help exit 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())

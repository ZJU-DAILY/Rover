#!/usr/bin/env python3

from pathlib import Path
import os
import subprocess
import sys


def main(argv=None):
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
    try:
        from rover_host import main as host_main

        return host_main(argv)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print('Rover: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

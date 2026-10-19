#!/usr/bin/env python3

from pathlib import Path
import sys


def main(argv=None):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Rover_walkthrough import main as walkthrough_main

    return walkthrough_main(['push', *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    raise SystemExit(main())

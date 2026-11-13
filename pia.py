"""Run or resume the configured 1,000-video PIA assessment."""

import sys

from scripts.seepat_cli import main

if __name__ == "__main__":
    raise SystemExit(main(["pia", *sys.argv[1:]]))

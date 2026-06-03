"""Thin wrapper so the suite can be run from the repository root."""

from interference_suite.run import main


if __name__ == "__main__":
    raise SystemExit(main())

"""Allow running ``python -m lzero.entry.async_split <command> ...``."""

from .cli import main

if __name__ == '__main__':
    raise SystemExit(main())

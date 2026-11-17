"""Portable local configuration; dotenv values never execute shell code."""
from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def load_environment(path: Path = PROJECT / '.env') -> None:
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        key, separator, value = line.partition('=')
        key = key.strip().removeprefix('export ')
        if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError(f'{path.name}:{number}: invalid environment assignment')
        parts = shlex.split(value, comments=True)
        os.environ.setdefault(key, ' '.join(parts))


def configured_path(name: str, default: str) -> Path:
    value = Path(os.environ.get(name) or default).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT / value).resolve()


load_environment()
ARDY_ROOT = configured_path('ARDY_ROOT', 'external/ardy')
MOTION_TOOLS = configured_path('MOTION_TOOLS_ROOT', '.local-data')
EMBEDDINGS = configured_path('ARDY_EMBEDDINGS', str(MOTION_TOOLS / 'embeddings.npz'))
SERVICE_DIR = configured_path('MOTION_SERVICE_DIR', 'public/service')

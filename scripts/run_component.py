"""Run a project component with its configured Python; preserve stdio for MCP."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from settings import load_environment

COMPONENTS = {
    'service': ('ARDY_PYTHON', 'service.py', ['http']),
    'mcp': ('MOTION_MCP_PYTHON', 'mcp_server.py', []),
    'perception': ('MOTION_PERCEPTION_PYTHON', 'perception_service.py', []),
    'supermimic': ('MOTION_PERCEPTION_PYTHON', 'supermimic_service.py', []),
    'python': ('MOTION_PYTHON', None, []),
    'perception-python': ('MOTION_PERCEPTION_PYTHON', None, []),
}


def interpreter(variable: str) -> str:
    configured = os.environ.get(variable) or os.environ.get('MOTION_PYTHON')
    if configured:
        candidate = shutil.which(configured) or str((ROOT / configured).resolve())
        if not Path(candidate).is_file():
            raise ValueError(f'{variable} does not identify a Python executable')
        return candidate
    folders = ('.venv', '.venv-perception', '.venv-mcp')
    if variable == 'MOTION_PERCEPTION_PYTHON':
        folders = ('.venv-perception', '.venv')
    elif variable == 'MOTION_MCP_PYTHON':
        folders = ('.venv', '.venv-mcp')
    for folder in folders:
        candidate = ROOT / folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def main() -> None:
    load_environment()
    if len(sys.argv) < 2 or sys.argv[1] not in COMPONENTS:
        raise ValueError('Invalid component; choose one of: service, mcp, perception, supermimic, python, perception-python')
    variable, script, arguments = COMPONENTS[sys.argv[1]]
    target = [str(ROOT / 'server' / script)] if script else []
    command = [interpreter(variable), *target, *arguments, *sys.argv[2:]]
    os.chdir(ROOT)
    os.execv(command[0], command)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f'Roamind launch failed: {error}')

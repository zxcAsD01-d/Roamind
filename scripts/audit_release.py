"""Audit staged Git objects without printing potentially sensitive content."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRECTORIES = frozenset(('models', 'datasets', 'external', 'output', '.local-data',
                                  '.playwright-cli', 'node_modules', 'dist'))
PATTERNS = {
    'local home path': re.compile(rb'/(?:Users|home)/[^\s/"\'<>]+/'),
    'private key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'GitHub credential': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'API credential': re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b'),
    'personal email': re.compile(rb'\b[\w.+-]+@(?:qq|gmail|hotmail|outlook)[.]com\b'),
}


def excluded(name: str) -> bool:
    parts = Path(name).parts
    return (any(part in PRIVATE_DIRECTORIES or part.startswith('.venv') for part in parts)
            or name in ('mcp.json', '.env') or name.startswith('.env.') and name != '.env.example'
            or name.startswith(('public/locomotion/', 'public/motion-table/', 'public/social-actions/'))
            or name.lower().endswith(('.vrm', '.sqlite3', '.pyc', '.log')))


def audit() -> list[str]:
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=ROOT).split(b'\0')
    issues = []
    count = size = 0
    for entry in filter(None, entries):
        meta, filename = entry.split(b'\t', 1)
        mode, identifier, stage = meta.split()
        name = filename.decode('utf-8')
        if excluded(name) or mode not in (b'100644', b'100755') or stage != b'0':
            issues.append(f'{name}: excluded path or non-regular Git entry')
            continue
        data = subprocess.check_output(['git', 'cat-file', 'blob', identifier.decode()], cwd=ROOT)
        count += 1; size += len(data)
        if len(data) > 5 * 1024 * 1024:
            issues.append(f'{name}: file exceeds 5 MiB')
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                issues.append(f'{name}: {label}')
    if not count:
        issues.append('Git index is empty; stage the intended release first')
    print(f'Audited {count} staged files, {size / 1048576:.2f} MiB; {len(issues)} issues')
    return issues


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracked', action='store_true', help='Audit the staged Git index (default)')
    parser.parse_args()
    findings = audit()
    for finding in findings:
        print(finding, file=sys.stderr)
    sys.exit(1 if findings else 0)

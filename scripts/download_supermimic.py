"""Download a pinned, independently trained G1 SuperMimic policy; no pickle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REPO = 'pantelismanouselis/g1-supermimic-lafan1'
REVISION = '907a358eab1ced90fb09b3df83a67354e7c90e18'
HASHES = {
    'policy.safetensors': 'a1c409acf38bc11a216a7decb120e4986c587cdd3aa3d383319dfaed990404ef',
    'config.json': '2dc9e60bb86efc0e35bcac7a063bf6b683ad3d5ef3c6bd853bc814ced46bc2dc',
}
DEFAULT_MODEL = ROOT / 'models/supermimic/g1-lafan1-040000'


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def download_file(url: str, target: Path, expected: str | None = None) -> dict:
    if target.is_file() and expected and digest(target) == expected:
        return {'file': target.name, 'sha256': expected, 'bytes': target.stat().st_size}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    try:
        with urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
            count = 0
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > 64 * 1024 * 1024:
                    raise ValueError(f'Download exceeds size limit: {target.name}')
                stream.write(block)
        actual = digest(temporary)
        if expected and actual != expected:
            raise ValueError(f'SHA256 mismatch: {target.name}')
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {'file': target.name, 'sha256': actual, 'bytes': target.stat().st_size}


def download_policy(output: Path) -> dict:
    names = [*HASHES, 'README.md', 'LICENSE', 'LICENSE.code', 'NOTICE.md',
             'manifest.json', 'verification.json']
    records = [download_file(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',
                             output / name, HASHES.get(name)) for name in names]
    report = {'repo': REPO, 'revision': REVISION, 'official': False,
              'robot': 'unitree-g1-29dof', 'files': records,
              'license': 'MIT for publisher rights in weights; third-party source terms in NOTICE.md'}
    (output / 'download.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(download_policy(args.output), indent=2))


if __name__ == '__main__':
    main()

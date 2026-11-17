"""Pinned G1 geometry and one LAFAN1 reference for isolated tracking validation."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from urllib.request import urlopen

from download_supermimic import ROOT, DEFAULT_MODEL, download_file

REPO = 'lvhaidong/LAFAN1_Retargeting_Dataset'
REVISION = 'ce1572906efe6157840e8474d5a0d7aa87481e74'
OUTPUT = ROOT / '.local-data/supermimic'


def main() -> None:
    config = json.loads((DEFAULT_MODEL / 'config.json').read_text())
    tree_url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/robot_description/g1/meshes'
    with urlopen(tree_url, timeout=30) as response:
        meshes = json.load(response)
    records = [('robot_description/g1/g1_29dof_rev_1_0.urdf',
                config['assets_sha256']['g1_description/g1_29dof_rev_1_0.urdf'])]
    records += [(entry['path'], None) for entry in meshes if entry['type'] == 'file']
    records.append(('g1/walk1_subject1.csv', config['dataset']['motion_sha256']['walk1_subject1.csv']))

    def fetch(record: tuple[str, str | None]) -> dict:
        name, expected = record
        relative = name.removeprefix('robot_description/')
        result = download_file(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}',
                               OUTPUT / relative, expected)
        result['file'] = relative
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(fetch, records))
    report = {'repo': REPO, 'revision': REVISION, 'files': downloaded,
              'notice': 'LAFAN1 CC BY-NC-ND 4.0; robot description original Unitree terms apply. Research test only.'}
    (OUTPUT / 'download.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'files': len(downloaded), 'bytes': sum(row['bytes'] for row in downloaded)}))


if __name__ == '__main__':
    main()

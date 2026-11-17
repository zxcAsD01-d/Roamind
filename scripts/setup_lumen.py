"""Install pinned optional HAST resources without copying its Git or dependencies."""
from __future__ import annotations

import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PIN = json.loads((ROOT / 'vendor/HAST-SEC/UPSTREAM.json').read_text())
PREFIXES = ('models/skills/', 'models/trajectories/', 'models/body/', 'models/control/', 'configs/', 'licenses/')


def install() -> None:
    target = ROOT / 'external/HAST-SEC'
    if (target / 'models/skills/manifest.json').is_file():
        print('Existing HAST resources retained. Remove the directory explicitly to reinstall.')
        return
    url = f"https://codeload.github.com/sola1039/HAST-SEC/tar.gz/{PIN['revision']}"
    with urlopen(url, timeout=60) as response:
        data = response.read(128 * 1024 * 1024 + 1)
    if len(data) > 128 * 1024 * 1024:
        raise ValueError('HAST archive exceeds 128 MiB')
    unpacked = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for member in archive:
            relative = PurePosixPath(*PurePosixPath(member.name).parts[1:])
            if '..' in relative.parts or relative.is_absolute():
                raise ValueError('Unsafe archive path')
            name = str(relative)
            if not member.isfile() or not (name.startswith(PREFIXES) or name in ('LICENSE', 'THIRD_PARTY_NOTICES.txt')):
                continue
            if member.size > 64 * 1024 * 1024:
                raise ValueError('HAST resource exceeds 64 MiB')
            unpacked += member.size
            if unpacked > 256 * 1024 * 1024:
                raise ValueError('HAST resources exceed 256 MiB')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f'Cannot extract {name}')
            destination = target / name
            if not destination.resolve().is_relative_to(target.resolve()):
                raise ValueError('Resource path escapes the installation directory')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(stream.read())
    print('Installed HAST resources. Non-commercial research terms: external/HAST-SEC/LICENSE')


if __name__ == '__main__':
    install()

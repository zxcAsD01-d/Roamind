"""Only locally curated, published table clips are callable; no invented action names."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'public'


@lru_cache(maxsize=1)
def actions():
    catalog = ROOT/'social-actions/catalog.json'
    if not catalog.exists() or not (ROOT/'motion-table/table.json').is_file():
        return ()
    rows = json.loads(catalog.read_text())['actions']
    published = {entry.get('prompt') for entry in json.loads((ROOT/'motion-table/table.json').read_text())['entries']
                 if entry['role'] == 'action' and entry['duration'] >= 2}
    return tuple({'action': row['action'], 'label': row['label'], 'prompt': row['prompt'],
                  'variants': len(row['clips']), 'channel': 'full-body', 'nativeTiming': True,
                  'overlayLocomotion': False} for row in rows if row['prompt'] in published)


def find(action):
    return next((row for row in actions() if row['action'] == action), None)

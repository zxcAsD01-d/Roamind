"""Curate native, stationary social recordings into the existing full-body motion table."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from build_locomotion import DEFAULT_DATA, PROJECT, document, load_recording
from build_motion_table import checked

GROUPS = {
    'clap': ('鼓掌', ('*clapping*poses.npz', '*clap_*stageii.npz')),
    'talk': ('交谈手势', ('TCD_handMocap/*/Talking*poses.npz',)),
    'wave-left': ('左手挥手', ('KIT/*/wave_left*poses.npz',)),
    'wave-right': ('右手挥手', ('KIT/*/wave_right*poses.npz',)),
    'wave-both': ('双手招呼', ('KIT/*/wave_both*poses.npz',)),
    'disagree': ('表达不同意见', ('*gesture_etc-32-no-*poses.npz',)),
    'stretch': ('舒展身体', ('*over stretch*poses.npz', '*stretches*poses.npz')),
    'point': ('指向示意', ('TCD_handMocap/*/pointing*poses.npz',)),
    'eat-reference': ('进食参考动作', ('GRAB/*/*_eat_*stageii.npz',)),
    'drink-reference': ('饮水参考动作', ('GRAB/*/cup_drink*stageii.npz', 'GRAB/*/mug_drink*stageii.npz')),
}


def paths_for(patterns):
    paths = set()
    for pattern in patterns:
        paths.update(DEFAULT_DATA.glob(pattern) if '/' in pattern else DEFAULT_DATA.rglob(pattern))
    return sorted(paths)


def windows(recording, identity):
    count = len(recording['root'])
    for begin in range(0, count-60, 30):
        end = min(count-1, begin+100)
        clip = document(recording, begin, end, identity+f'_{begin}', 'action')
        if clip['distance'] > .55 or abs(clip['angle']) > .35:
            continue
        arms = recording['local'][begin:end+1, [8, 9, 14, 15]]
        rotations = Rotation.from_rotvec(arms.reshape(-1, 3)).as_quat().reshape(-1, 4, 4)
        angles = 2*np.arccos(np.clip(np.abs((rotations*rotations[:1]).sum(axis=2)), 0, 1))
        if angles.max() < .3:
            continue
        yield clip


def collect(output):
    report, catalog = [], []
    output.mkdir(parents=True, exist_ok=True)
    for key, (label, patterns) in GROUPS.items():
        accepted, sources = [], Counter()
        for path in paths_for(patterns)[:32]:
            try:
                recording = load_recording(path)
                identity = 'social_'+key+'_'+hashlib.sha256(str(path).encode()).hexdigest()[:8]
                for clip in windows(recording, identity):
                    if sources[recording['source']] >= 2:
                        break
                    try:
                        quality = checked(clip)
                        clip['prompt'] = 'AMASS/social/'+key
                        clip['quality'] = quality
                        (output/(clip['name']+'.json')).write_text(json.dumps(clip, separators=(',', ':'), allow_nan=False))
                        accepted.append({'file': clip['name'], 'source': clip['source'], 'quality': quality})
                        sources[recording['source']] += 1
                    except ValueError as error:
                        report.append({'source': recording['source'], 'frames': clip['sourceFrames'], 'reason': str(error)})
                    if len(accepted) >= 6:
                        break
            except (ValueError, KeyError, OSError) as error:
                report.append({'source': str(path.relative_to(DEFAULT_DATA)), 'reason': str(error)})
            if len(accepted) >= 6:
                break
        if accepted:
            catalog.append({'action': 'amass:'+key, 'label': label, 'prompt': 'AMASS/social/'+key, 'clips': accepted})
        print(json.dumps({'action': key, 'accepted': len(accepted)}, ensure_ascii=False), flush=True)
    public = PROJECT/'public/social-actions'
    public.mkdir(parents=True, exist_ok=True)
    payload = {'version': 1, 'actions': catalog, 'rejected': report,
               'license': 'Local AMASS and contributing dataset terms apply; do not redistribute without authorization.',
               'policy': 'native-joints-and-root; source support gates; avatar and seam gates at playback; eat/drink references do not satisfy needs'}
    (public/'catalog.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({'categories': len(catalog), 'clips': sum(len(row['clips']) for row in catalog), 'rejected': len(report)}))


if __name__ == '__main__':
    collect(PROJECT/'.local-data/table-candidates')

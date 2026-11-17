"""Discover executable arm tracks from the installed HAST resource manifest."""
from functools import lru_cache
import json
from pathlib import Path
from settings import configured_path

ROOT = configured_path('MOTION_LUMEN_DIR', 'external/HAST-SEC')
ALIASES = {'wave': 'right-摆动手臂', 'raise': 'right-向上举过头部', 'point': 'right-向前移动'}


@lru_cache(maxsize=1)
def native_actions():
    path = ROOT/'models/skills/manifest.json'
    if not path.is_file():
        return ()
    manifest = json.loads(path.read_text())
    bones = manifest['bones']
    if manifest['rate'] != 20 or len(bones) != 53:
        raise ValueError('流明动作目录骨架不兼容')
    allowed = {side+part for side in ('left', 'right')
               for part in ('Shoulder', 'UpperArm', 'LowerArm', 'Hand')}
    return tuple(row['key'] for row in manifest['classes']
                 if row['mask'] and all(type(i) is int and 0 <= i < len(bones)
                                        and bones[i] in allowed for i in row['mask']))


def supports(action):
    if not isinstance(action, str):
        return False
    key = ALIASES.get(action, action.removeprefix('lumen:') if action.startswith('lumen:') else '')
    return bool(key) and key in native_actions()


def side(action):
    if not supports(action):
        raise ValueError('流明动作不存在')
    return ALIASES.get(action, action.removeprefix('lumen:')).split('-', 1)[0]


def capabilities():
    from social_actions import actions
    return {'defaultUpperBodyEngine': 'HAST-SEC/beatLayers+outputSmoothing',
            'durationSeconds': {'min': 1.5, 'max': 7.5},
            'actions': [{'action': 'lumen:'+key, 'label': key, 'channel': 'arms',
                         'overlayLocomotion': True} for key in native_actions()]+list(actions()),
            'nodeActions': [{'operation': operation, 'variants': 3, 'requires': 'fresh visual grasp + executable prop + mouth contact'}
                           for operation in ('eat', 'drink')],
            'aliases': dict(ALIASES), 'locomotionEngine': 'native-AMASS-root-motion',
            'contactEngine': 'visual-target+ARDY+bounded-chain-tracking',
            'license': 'HAST-SEC 非商业学习与研究许可证 0.1',
            'limits': ['上身动作不控制根与腿', '手臂前移动作不等于几何精确指向', '资源目录需保留 external/HAST-SEC']}

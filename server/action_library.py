"""Persistent, frequency-sized motion variants with quality-gated playback."""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import library_motion as motion_math

DEFAULT_DB = Path(__file__).resolve().parents[1]/'.local-data/motion-library.sqlite3'
MAX_VARIANTS = 4


class ActionLibrary:
    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        self.lock = threading.RLock()

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, sqlite3.connect(self.path, timeout=10) as db:
            db.row_factory = sqlite3.Row
            db.execute('CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, action TEXT, uses INTEGER DEFAULT 0, pending REAL DEFAULT 0, error TEXT DEFAULT "")')
            db.execute('CREATE TABLE IF NOT EXISTS clips (id TEXT PRIMARY KEY, key TEXT, action TEXT, duration REAL, motion TEXT, quality TEXT, plays INTEGER DEFAULT 0, last_play REAL DEFAULT 0, created REAL)')
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise

    @staticmethod
    def target(uses):
        return min(MAX_VARIANTS, 1+math.isqrt(max(0, uses-1))//2)

    def resolve(self, plan, history, clip_id=None):
        key, _, _ = motion_math.profile(plan, history)
        action = '+'.join(step['action'] for step in plan['timeline'])
        with self.connection() as db:
            db.execute('INSERT OR IGNORE INTO requests (key,action) VALUES (?,?)', (key, action))
            db.execute('UPDATE requests SET uses=uses+1 WHERE key=?', (key,))
            request = db.execute('SELECT * FROM requests WHERE key=?', (key,)).fetchone()
            entries = db.execute('SELECT * FROM clips WHERE key=? ORDER BY last_play,plays,id', (key,)).fetchall()
            if clip_id and not any(entry['id'] == clip_id for entry in entries):
                raise ValueError('指定变体与当前动作、时长或轨迹不匹配')
            target = self.target(request['uses'])
            expand = not clip_id and len(entries) < target and time.time()-request['pending'] > 60
            if not entries or expand:
                db.execute('UPDATE requests SET pending=? WHERE key=?', (time.time(), key))
                return {'hit': False, 'reason': 'missing' if not entries else 'expand',
                        'variants': len(entries), 'targetVariants': target, 'uses': request['uses']}
            candidates = [entry for entry in entries if not clip_id or entry['id'] == clip_id]
            if clip_id and not candidates:
                raise ValueError('指定变体与当前动作、时长或轨迹不匹配')
            return self.choose(db, candidates, plan, history, target)

    def choose(self, db, entries, plan, history, target):
        latest = max(entries, key=lambda entry: entry['last_play'])['id'] if len(entries) > 1 else None
        ranked = sorted(entries, key=lambda entry: (entry['id'] == latest, self.entry_cost(entry, history)))
        errors = []
        for entry in ranked:
            try:
                motion = motion_math.replay(json.loads(entry['motion']), plan, history)
            except ValueError as exc:
                errors.append(f"{entry['id']}: {exc}")
                continue
            db.execute('UPDATE clips SET plays=plays+1,last_play=? WHERE id=?', (time.time(), entry['id']))
            return {'hit': True, 'clipId': entry['id'], 'motion': motion, 'variants': len(entries),
                    'targetVariants': target, 'quality': json.loads(entry['quality'])}
        return {'hit': False, 'reason': 'unsafe_or_incompatible', 'detail': '; '.join(errors),
                'variants': len(entries), 'targetVariants': target}

    @staticmethod
    def entry_cost(entry, history):
        # Keep the newest playback out of the first choice while matching entry pose.
        if history is None:
            return entry['last_play']
        local, _ = motion_math.arrays(history)
        angle = float(motion_math.yaw(local[-1]))
        current = local[-1].copy()
        current[0] = (Rotation.from_euler('y', -angle)*Rotation.from_rotvec(current[0])).as_rotvec()
        target = np.asarray(json.loads(entry['motion'])['localRot'][:81]).reshape(27, 3)
        cost = (Rotation.from_rotvec(current).inv()*Rotation.from_rotvec(target)).magnitude()
        recent = max(0, 1-(time.time()-entry['last_play'])/120)
        return float(np.square(cost).mean())+recent*2+entry['plays']*.005

    def remember(self, plan, motion, prefix, history):
        key, _, _ = motion_math.profile(plan, history)
        try:
            clip = motion_math.canonical(plan, motion, prefix, history)
            quality = motion_math.quality(clip)
        except (ValueError, KeyError) as exc:
            return self.reject(key, str(exc))
        action = '+'.join(step['action'] for step in plan['timeline'])
        identity = f'clip_{uuid.uuid4().hex[:12]}'
        with self.connection() as db:
            existing = db.execute('SELECT id,motion FROM clips WHERE key=?', (key,)).fetchall()
            if any(self.duplicate(clip, json.loads(entry['motion'])) for entry in existing):
                db.execute('UPDATE requests SET pending=0,error=? WHERE key=?', ('生成结果与已有变体重复', key))
                return {'accepted': False, 'reason': '生成结果与已有变体重复'}
            if len(existing) >= MAX_VARIANTS:
                return {'accepted': False, 'reason': '此动作条件的变体数已达上限'}
            db.execute('INSERT INTO clips VALUES (?,?,?,?,?,?,0,?,?)',
                       (identity, key, action, clip['duration'], json.dumps(clip, separators=(',', ':')),
                        json.dumps(quality), time.time(), time.time()))
            db.execute('UPDATE requests SET pending=0,error="" WHERE key=?', (key,))
            db.execute('DELETE FROM clips WHERE id IN (SELECT id FROM clips ORDER BY plays,last_play LIMIT MAX(0,(SELECT COUNT(*) FROM clips)-512))')
        return {'accepted': True, 'clipId': identity, 'variants': len(existing)+1, 'quality': quality}

    @staticmethod
    def duplicate(first, second):
        a, b = np.asarray(first['localRot']).reshape(-1, 3), np.asarray(second['localRot']).reshape(-1, 3)
        if a.shape != b.shape:
            return False
        angles = (Rotation.from_rotvec(a).inv()*Rotation.from_rotvec(b)).magnitude().reshape(-1, 27)
        return float(angles.mean()) < .025 and float(angles.mean(axis=0).max()) < .08

    def reject(self, key, reason):
        with self.connection() as db:
            db.execute('UPDATE requests SET pending=0,error=? WHERE key=?', (reason, key))
        return {'accepted': False, 'reason': reason}

    def catalog(self):
        with self.connection() as db:
            clips = [dict(row) for row in db.execute('SELECT id,action,duration,quality,plays,created FROM clips ORDER BY action,created')]
            requests = [dict(row) for row in db.execute('SELECT action,SUM(uses) uses,MAX(error) error FROM requests GROUP BY action')]
        for entry in clips:
            entry['quality'] = json.loads(entry['quality'])
        return {'clips': clips, 'usage': requests, 'maxVariants': MAX_VARIANTS,
                'policy': 'missing_or_frequency_expand', 'variantUseThresholds': [1, 5, 17, 37]}

    def clip(self, identity):
        with self.connection() as db:
            row = db.execute('SELECT motion FROM clips WHERE id=?', (identity,)).fetchone()
        if row is None:
            raise ValueError('库内动作不存在')
        return json.loads(row['motion'])

    def describe(self, identity):
        with self.connection() as db:
            row = db.execute('SELECT id,action,duration FROM clips WHERE id=?', (identity,)).fetchone()
        if row is None:
            raise ValueError('库内动作不存在')
        return dict(row)


LIBRARY = ActionLibrary(os.environ.get('MOTION_LIBRARY_DB', DEFAULT_DB))


def remember_generated(plan, motion, prefix, history):
    try:
        return LIBRARY.remember(plan, motion, prefix, history)
    except (OSError, sqlite3.Error):
        logging.exception('生成动作已完成，但动作库保存失败')
        return {'accepted': False, 'reason': '动作库保存失败；详见动作服务日志'}

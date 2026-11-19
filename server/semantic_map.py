"""Bounded voxel and object memory in the visually estimated world frame."""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict

from map_geometry import MapGeometry, pack_keys
from stereo_depth import free_space_conflicts

import cv2
import numpy as np


class SemanticMap:
    environment_age = 1800
    person_age = 5
    dynamic_labels = {'person', 'human', 'face'}
    def __init__(self, voxel_size=.08, limit=40000):
        self.voxel_size, self.limit = voxel_size, limit
        self.voxels = OrderedDict()
        self.objects = {}
        self.surface_segments = OrderedDict()
        self.volumes = OrderedDict()
        self.labels = ['unlabeled']
        self.revision = 0
        self.map_id = uuid.uuid4().hex
        self.center = np.zeros(3)
        self.last_expire = float('-inf')
        self.expire_center = np.zeros(3)
        self._geometry_dirty = True
        self._geometry_updated = float('-inf')
        self._geometry_interval = .35
        self.geometry = MapGeometry(voxel_size)
        self.geometry_ms = 0.
        self._snapshot_cache = None
        self._last_visibility_check = float('-inf')
        self._cleared_at = {}

    @staticmethod
    def world_points(points, pose):
        return points@pose[:3, :3].T+pose[:3, 3]

    def integrate(self, frame, pose, timestamp):
        points, colors, rows, cols = frame.mapping_points(stride=4)
        supported = frame.valid if frame.surface_valid is None else frame.surface_valid
        self._merge(self.world_points(points, pose), colors, supported[rows, cols], timestamp)
        self._clear_weak_voxels(frame, pose, timestamp)
        self._expire(timestamp, pose[:3, 3])
        self.center = pose[:3, 3].copy()
        self.revision += 1

    def _clear_weak_voxels(self, frame, pose, timestamp):
        if timestamp <= self._last_visibility_check:
            return
        self._last_visibility_check = timestamp
        # Annotations may still arrive for earlier frames; don't resurrect cleared points.
        self._cleared_at = {key: stamp for key, stamp in self._cleared_at.items() if timestamp-stamp <= 4.}
        candidates = [(key, voxel) for key, voxel in self.voxels.items()
                      if voxel['measured'] and voxel['frames'] == 1 and voxel['lastSeen'] < timestamp]
        if not candidates:
            return
        points = np.asarray([voxel['point'] for _, voxel in candidates])
        optical = (points-pose[:3, 3])@pose[:3, :3]
        conflicts = free_space_conflicts(frame, optical, self.voxel_size)
        removed = []
        for (key, voxel), conflict in zip(candidates, conflicts):
            if not conflict:
                voxel.pop('freeSpaceVotes', None)
            elif voxel.get('freeSpaceVotes', 0) == 1:
                self._cleared_at[key] = timestamp
                del self.voxels[key]
                removed.append(key)
            else:
                voxel['freeSpaceVotes'] = 1
        if removed:
            self._invalidate_removed(removed, refresh=True)

    def _merge(self, world, colors, measured, timestamp, label=0, confidence=0., instance=''):
        """Reduce observations and update geometry statistics in NumPy batches."""
        if not len(world):
            return []
        keys, points, rgb, direct = self._reduce(world, colors, measured)
        if self._cleared_at:
            keep = [self._cleared_at.get(key, float('-inf')) < timestamp for key in keys]
            keys = [key for key, retain in zip(keys, keep) if retain]
            if not keys:
                return []
            points, rgb, direct = points[keep], rgb[keep], direct[keep]
        records = [self.voxels.get(key) for key in keys]
        existing = np.array([i for i, record in enumerate(records) if record is not None], dtype=int)
        if len(existing):
            self._update_existing(existing, records, points, rgb, direct, timestamp)
        normal_keys, changed = [], False
        for i, key in enumerate(keys):
            voxel = records[i]
            if voxel is None:
                voxel = {'point': points[i], 'color': rgb[i], 'count': 1, 'frames': 1, 'label': 0,
                         'labelConfidence': 0., 'measured': bool(direct[i]), 'lastSeen': timestamp}
                self.voxels[key] = voxel
                normal_keys.append(key)
                changed = True
            elif voxel.pop('_updated', False):
                changed = True
                if voxel.pop('_normalChanged', False):
                    normal_keys.append(key)
            if label and self._write_label(voxel, label, confidence, timestamp, instance):
                changed = True
            if voxel['lastSeen'] == timestamp:
                self.voxels.move_to_end(key)
        if changed:
            self.geometry.changed(normal_keys)
            self._geometry_dirty = True
            self._snapshot_cache = None
        return keys

    def _reduce(self, world, colors, measured):
        raw_keys = np.floor(world/self.voxel_size).astype(np.int64)
        _, first, inverse = np.unique(pack_keys(raw_keys), return_index=True, return_inverse=True)
        count = len(first)
        direct = np.bincount(inverse, weights=measured, minlength=count) > 0
        weights = measured | ~direct[inverse]
        counts = np.bincount(inverse, weights=weights, minlength=count)
        points = np.column_stack([np.bincount(inverse, weights=world[:, axis]*weights)/counts
                                  for axis in range(3)])
        rgb = np.column_stack([np.bincount(inverse, weights=colors[:, axis]*weights)/counts
                               for axis in range(3)])
        keys = [tuple(key) for key in raw_keys[first].tolist()]
        return keys, points, rgb, direct

    @staticmethod
    def _update_existing(indices, records, points, rgb, direct, timestamp):
        old = [records[i] for i in indices]
        seen = np.array([v['lastSeen'] for v in old])
        measured = np.array([v.get('measured', True) for v in old])
        replace = direct[indices] & ~measured & (timestamp >= seen)
        update = (timestamp > seen) & (~measured | direct[indices])
        accepted = update | replace
        counts = np.array([v['count'] for v in old])
        counts = np.where(replace, 1, np.minimum(20, counts+1))
        old_points = np.asarray([v['point'] for v in old])
        old_colors = np.asarray([v['color'] for v in old])
        merged_points = old_points+(points[indices]-old_points)/counts[:, None]
        merged_colors = old_colors+(rgb[indices]-old_colors)/counts[:, None]
        merged_points[replace], merged_colors[replace] = points[indices[replace]], rgb[indices[replace]]
        reference = np.asarray([v.get('normalPoint', v['point']) for v in old])
        normal_changed = (np.linalg.norm(merged_points-reference, axis=1) > .004) | replace
        for local in np.flatnonzero(accepted):
            voxel = old[local]
            voxel.update(point=merged_points[local], color=merged_colors[local], count=int(counts[local]),
                         frames=1 if replace[local] else min(255, voxel['frames']+1),
                         measured=bool(measured[local] or direct[indices[local]]),
                         lastSeen=timestamp, _updated=True, _normalChanged=bool(normal_changed[local]))

    @staticmethod
    def _write_label(voxel, label, confidence, timestamp, instance):
        previous = max(voxel.get('labelSeen', float('-inf')), voxel.get('labelEvidenceSeen', float('-inf')))
        if timestamp < previous or (timestamp == previous and confidence <= voxel.get('labelConfidence', 0.)):
            return False
        # A newer weak one-off prediction cannot replace an accumulated label.
        votes = voxel.setdefault('labelVotes', {})
        if not votes and voxel['label']:
            votes[voxel['label']] = voxel.get('labelConfidence', 0.)
        if timestamp > previous:
            votes.update({key: value*.98 for key, value in votes.items()})
            votes[label] = min(20., votes.get(label, 0.)+confidence)
        elif label == voxel['label']:
            votes[label] = max(votes.get(label, 0.), confidence)
        else:
            votes[label] = confidence
        winner = max(votes, key=votes.get)
        voxel['labelEvidenceSeen'] = timestamp
        if len(votes) > 4:
            del votes[min(votes, key=votes.get)]
        if winner != label:
            return False
        if voxel['label'] != label or (instance and instance != voxel.get('instanceId')):
            voxel.pop('semanticNormal', None)
        voxel.update(label=int(label), labelSeen=timestamp, labelConfidence=confidence,
                     instanceId=instance or voxel.get('instanceId', ''), inherited=False)
        return True

    def _ensure_geometry(self, force=False):
        if not self._geometry_dirty:
            return
        if not force and self.voxels and time.monotonic()-self._geometry_updated < self._geometry_interval:
            return
        started = time.perf_counter()
        self.geometry.rebuild(self.voxels, self.labels, self.dynamic_labels)
        self.surface_segments, self.volumes = self.geometry.surfaces, self.geometry.volumes
        self._geometry_dirty = False
        self._geometry_updated = time.monotonic()
        self.geometry_ms = round((time.perf_counter()-started)*1000, 3)
        self._snapshot_cache = None
        self.revision += 1

    def _expire(self, timestamp, center):
        moved = np.linalg.norm(center-self.expire_center) > .5
        removed, expired_geometry = [], False
        if timestamp-self.last_expire >= .5 or moved:
            keys, values = list(self.voxels), list(self.voxels.values())
            if values:
                points = np.asarray([voxel['point'] for voxel in values])
                seen = np.fromiter((voxel['lastSeen'] for voxel in values), float)
                dynamic = np.array([self.labels[v['label']] in self.dynamic_labels for v in values])
                limits = np.where(dynamic, self.person_age, self.environment_age)
                expired = (timestamp-seen > limits) | (np.sum((points-center)**2, axis=1) > 18**2)
                removed = [keys[index] for index in np.flatnonzero(expired)]
                expired_geometry = bool(removed)
                for key in removed:
                    del self.voxels[key]
            self.last_expire, self.expire_center = timestamp, center.copy()
        while len(self.voxels) > self.limit:
            key, _ = self.voxels.popitem(last=False)
            removed.append(key)
        if removed:
            self._invalidate_removed(removed, refresh=expired_geometry)
        objects = {key: item for key, item in self.objects.items()
                   if timestamp-item['lastSeen'] < (300 if item['label'] in self.dynamic_labels else self.environment_age)}
        if len(objects) != len(self.objects):
            self.revision += 1
        self.objects = objects

    def _invalidate_removed(self, removed, refresh):
        self.geometry.removed(removed)
        self._geometry_dirty = True
        if refresh:
            self._geometry_updated = float('-inf')
        self._snapshot_cache = None
        self.revision += 1

    @staticmethod
    def surface(frame, box):
        """Depth-connected surface inside the box, excluding background and edges."""
        x0, y0, x1, y1 = np.rint(box).astype(int)
        height, width = frame.depth.shape
        x0, x1 = np.clip([x0, x1], 0, width)
        y0, y1 = np.clip([y0, y1], 0, height)
        if x1-x0 < 5 or y1-y0 < 5:
            return None
        depth = frame.reconstruction_depth
        depth = frame.depth if depth is None else depth
        supported = frame.valid if frame.surface_valid is None else frame.surface_valid
        supported = supported if frame.reconstruction_depth is None else depth > 0
        crop = depth[y0:y1, x0:x1]
        valid = supported[y0:y1, x0:x1] & (frame.edges[y0:y1, x0:x1] == 0)
        values = crop[valid]
        if len(values) < 20:
            return None
        # A box often contains more background than object. Seed depth from its
        # central region, then grow only a supported connected surface.
        inset_x, inset_y = max(1, (x1-x0)//4), max(1, (y1-y0)//4)
        core = crop[inset_y:-inset_y, inset_x:-inset_x]
        core_valid = valid[inset_y:-inset_y, inset_x:-inset_x]
        seeds = core[core_valid]
        if len(seeds) < 20:
            return None
        bins = np.floor(seeds/.2).astype(int)
        peak = np.bincount(bins).argmax()
        depth = float(np.median(seeds[bins == peak]))
        mask = (valid & (np.abs(crop-depth) < max(.2, depth*.06))).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return None
        core_components = components[inset_y:-inset_y, inset_x:-inset_x]
        votes = np.bincount(core_components.ravel(), minlength=count)
        votes[0] = 0
        component = int(np.argmax(votes))
        rows, cols = np.nonzero(components == component)
        if len(rows) < 20:
            return None
        return rows+y0, cols+x0

    def annotate(self, frame, pose, detections, timestamp):
        associations, used = [], set()
        for detection in detections[:32]:
            parsed = self._detection(detection)
            if parsed is not None:
                association = self._associate(frame, pose, *parsed, timestamp, used)
                if association is not None:
                    associations.append(association)
                    used.add(association['id'])
        self._expire(max(timestamp, self.last_expire), self.center)
        if associations:
            self._ensure_geometry(force=True)
        for association in associations:
            keys = association.pop('_voxelKeys')
            ids = {self.voxels[key].get('volumeId') for key in keys if key in self.voxels
                   and self.voxels[key].get('instanceId') == association['id']}
            ids.discard(None)
            association['volumeIds'] = sorted(ids)
            association['surfaceIds'] = sorted({sid for vid in ids
                for sid in self.volumes.get(vid, {}).get('surfaceIds', [])})
        self.revision += 1
        return associations

    @staticmethod
    def _detection(detection):
        if not isinstance(detection, dict):
            return None
        box, label = detection.get('box'), detection.get('label')
        if not isinstance(box, list) or len(box) != 4 or not isinstance(label, str) or not 1 <= len(label) <= 80:
            return None
        try:
            box = [float(value) for value in box]
            confidence = float(detection.get('confidence', 0))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(box).all() or not np.isfinite(confidence) or not 0 <= confidence <= 1:
            return None
        return box, label, confidence

    def _associate(self, frame, pose, box, label, confidence, timestamp, used):
        surface = self.surface(frame, box)
        if surface is None or (label not in self.labels and len(self.labels) >= 256):
            return None
        rows, cols = surface
        depth = frame.depth if frame.reconstruction_depth is None else frame.reconstruction_depth
        z, c = depth[rows, cols], frame.calibration
        optical = np.column_stack(((cols-c.cx)*z/c.fx, (rows-c.cy)*z/c.fy, z))
        world = self.world_points(optical, pose)
        if label not in self.labels:
            self.labels.append(label)
        supported = frame.valid if frame.surface_valid is None else frame.surface_valid
        measured = supported[rows, cols]
        obj = self._object(label, world, confidence, timestamp, used)
        sample = (rows % 3 == 0) & (cols % 3 == 0)
        keys = self._merge(world[sample], frame.rgb[rows[sample], cols[sample], ::-1], measured[sample],
                           timestamp, self.labels.index(label), confidence, obj['id'])
        if obj['lastSeen'] == timestamp:
            obj['inferredFraction'] = round(float((~measured).mean()), 3)
            obj['geometry'] = 'measured-and-plane-inferred-surface' if not measured.all() else 'visible-surface-only'
        pixel_index = self._surface_pixel(box, rows, cols, measured)
        direct_world = world[measured]
        sigma = self.position_uncertainty(z[measured], c)
        return {'id': obj['id'], 'label': label, 'points': len(world),
                'worldPosition': np.median(direct_world, axis=0).round(4).tolist() if len(direct_world) else obj['center'],
                'positionSigmaMeters': sigma, 'confidence': confidence,
                'surfacePixel': [int(cols[pixel_index]), int(rows[pixel_index])],
                'depth': round(float(np.median(z)), 3), 'box': box,
                'measuredPoints': int(measured.sum()), 'inferredPoints': int((~measured).sum()),
                'boundary': self.boundary(frame.depth.shape, rows, cols), '_voxelKeys': keys}

    @staticmethod
    def position_uncertainty(depth, calibration):
        if not len(depth):
            return 1.
        median = float(np.median(depth))
        # Half-pixel disparity uncertainty plus the robust surface depth spread.
        quantization = .5*median**2/(calibration.fx*calibration.baseline)
        spread = float(np.median(np.abs(depth-median)))*1.4826
        return round(max(.005, quantization, spread), 4)

    @staticmethod
    def _surface_pixel(box, rows, cols, measured):
        distances = (cols-(box[0]+box[2])/2)**2+(rows-(box[1]+box[3])/2)**2
        if measured.any():
            distances = np.where(measured, distances, np.inf)
        return int(np.argmin(distances))

    @staticmethod
    def boundary(shape, rows, cols):
        mask = np.zeros(shape, np.uint8)
        mask[rows, cols] = 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        simplified = cv2.approxPolyDP(contour, .004*cv2.arcLength(contour, True), True)
        return simplified.reshape(-1, 2)[:128].tolist()

    def _object(self, label, points, confidence, timestamp, excluded):
        low, high = np.percentile(points, [5, 95], axis=0)
        center = np.median(points, axis=0)
        candidates = [item for item in self.objects.values() if item['label'] == label and item['id'] not in excluded
                      and np.linalg.norm(np.asarray(item['center'])-center) < max(.5, np.linalg.norm(high-low)*.5)]
        previous = min(candidates, key=lambda item: np.linalg.norm(np.asarray(item['center'])-center)) if candidates else None
        if previous is not None and timestamp <= previous['lastSeen']:
            return previous
        identity = previous['id'] if previous else f'object-{uuid.uuid4().hex[:10]}'
        item = {'id': identity, 'label': label, 'center': center.round(3).tolist(),
                'bounds': [low.round(3).tolist(), high.round(3).tolist()], 'confidence': round(confidence, 3),
                'observations': 1 if previous is None else previous['observations']+1,
                'lastSeen': timestamp, 'source': 'yolo-box-depth-connected-surface',
                'geometry': 'visible-surface-only'}
        self.objects[identity] = item
        return item

    def planning_points(self, now):
        """Measured fused voxels before any rendering stride or surface completion."""
        self._expire(now, self.center)
        return {'frame': 'visual-odometry', 'mapId': self.map_id, 'revision': self.revision,
                'voxelSize': self.voxel_size,
                'measuredPoints': [v['point'].tolist() for v in self.voxels.values() if v['measured']]}

    def snapshot(self, timestamp=None):
        now = time.time() if timestamp is None else timestamp
        self._expire(now, self.center)
        self._ensure_geometry()
        if self._snapshot_cache is None:
            self._snapshot_cache = self._cloud_payload()
        return {**self._snapshot_cache, 'mapId': self.map_id, 'revision': self.revision,
                'geometryRevision': self.geometry.revision,
                'objects': [{**item, 'ageSeconds': round(max(0, now-item['lastSeen']), 2),
                             'visible': now-item['lastSeen'] < 2} for item in self.objects.values()]}

    def _cloud_payload(self):
        values = list(self.voxels.values())
        stride = max(1, int(np.ceil(len(values)/6000)))
        visible = values[::stride]
        people = [voxel for voxel in visible if self.labels[voxel['label']] in self.dynamic_labels]
        colors = [v.get('surfaceColor') for v in visible]
        if visible:
            raw = np.clip(np.rint([v['color'] for v in visible]), 0, 255).astype(int)
            packed = (raw[:, 0] << 16) | (raw[:, 1] << 8) | raw[:, 2]
            colors = [int(packed[i]) if color is None else color for i, color in enumerate(colors)]
        return {'voxelSize': self.voxel_size, 'totalPoints': len(values),
                'inferredPoints': sum(not voxel.get('measured', True) for voxel in values),
                'positions': self._positions(visible), 'colors': colors,
                'semantics': [v['label'] for v in visible], 'labels': self.labels.copy(),
                'volumeIds': [v.get('volumeId') for v in visible],
                'inferred': [not voxel.get('measured', True) for voxel in visible],
                'surfaces': list(self.surface_segments.values()), 'volumes': list(self.volumes.values()),
                'personCloud': {'positions': self._positions(people),
                                'semantics': [voxel['label'] for voxel in people], 'maxAgeSeconds': self.person_age},
                'frame': 'gravity-aligned-y-up', 'persistent': True,
                'semanticPolicy': 'same-frame-yolo-instance-votes; inherit-only-compatible-connected-surfaces',
                'geometry': {'surfaceCount': len(self.surface_segments), 'volumeCount': len(self.volumes),
                             'rebuildMs': self.geometry_ms, 'usage': 'reconstruction-only',
                             'surfaceColorPolicy': 'measured-median-rgb; temporal-step-limit-4',
                             'volumePolicy': 'normal-color-instance-connected-voxels'},
                'limits': {'maxVoxels': self.limit, 'maxAgeSeconds': self.environment_age,
                           'maxSurfaces': self.geometry.max_surfaces, 'maxVolumes': self.geometry.max_volumes,
                           'maxSurfaceQuads': self.geometry.max_quads,
                           'personPointAgeSeconds': self.person_age, 'personMemorySeconds': 300, 'radiusMeters': 18}}

    @staticmethod
    def _positions(voxels):
        return np.asarray([v['point'] for v in voxels]).round(3).ravel().tolist() if voxels else []

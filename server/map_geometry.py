"""Incremental local normals, supported surface tiles and semantic components.

Geometry is a reconstruction layer. It never supplies free space or collision
evidence. A tile covers an occupied voxel footprint, never a component hull.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ZERO = np.zeros(3)

def pack_keys(keys):
    """Pack signed voxel coordinates; the 18 m map uses a tiny part of this range."""
    shifted = np.asarray(keys, dtype=np.int64) + (1 << 20)
    if np.any((shifted < 0) | (shifted >= 1 << 21)):
        raise ValueError('点云坐标超出体素索引范围')
    return (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]


def packed_color(color):
    rgb = np.clip(np.rint(color), 0, 255).astype(int)
    return int(rgb[0] << 16 | rgb[1] << 8 | rgb[2])


def mean_normal(normals):
    aligned = normals*np.where(normals@normals[0] < 0, -1., 1.)[:, None]
    normal = aligned.mean(axis=0)
    return normal/max(1e-9, np.linalg.norm(normal))


def component_groups(count, left, right):
    graph = coo_matrix((np.ones(len(left), np.uint8), (left, right)), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind='stable')
    return np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)


class MapGeometry:
    block_width = 8
    max_surfaces = 512
    max_volumes = 512
    max_quads = 6000

    def __init__(self, voxel_size):
        self.size = voxel_size
        self.blocks = defaultdict(set)
        self.dirty = set()
        self.surfaces = {}
        self.volumes = {}
        self.revision = 0
        self._serial = 0

    def changed(self, keys):
        for key in keys:
            block = tuple(value // self.block_width for value in key)
            self.blocks[block].add(key)
            self.dirty.add(key)

    def removed(self, keys):
        for key in keys:
            block = tuple(value // self.block_width for value in key)
            self.blocks[block].discard(key)
            if not self.blocks[block]:
                del self.blocks[block]
            self.dirty.discard(key)

    def _refresh_normals(self, voxels):
        core = sorted(self.dirty)
        if not core:
            return
        invalid = [key for key, value in voxels.items()
                   if not value.get('normalValid', False) and key not in self.dirty]
        if invalid:
            observed = np.asarray([voxels[key]['point'] for key in core])
            unsupported = np.asarray([voxels[key]['point'] for key in invalid])
            distance, _ = cKDTree(observed).query(unsupported, distance_upper_bound=self.size*2.8)
            core.extend(key for key, nearby in zip(invalid, np.isfinite(distance)) if nearby)
        points = np.asarray([voxel['point'] for voxel in voxels.values()])
        self._local_normals(core, voxels, cKDTree(points), points)

    def _local_normals(self, core, voxels, tree, points):
        centers = np.asarray([voxels[key]['point'] for key in core])
        distances, indices = tree.query(centers, k=min(12, len(points)))
        if distances.ndim == 1:
            for key in core:
                voxels[key]['normal'] = np.zeros(3)
                voxels[key]['normalValid'] = False
            return
        keep = distances <= self.size*2.8
        neighbors = points[indices]
        count = keep.sum(axis=1)
        mean = np.sum(neighbors*keep[:, :, None], axis=1)/np.maximum(1, count[:, None])
        centered = (neighbors-mean[:, None, :])*keep[:, :, None]
        covariance = np.einsum('nki,nkj->nij', centered, centered)/np.maximum(1, count[:, None, None])
        values, vectors = np.linalg.eigh(covariance)
        normals = vectors[:, :, 0]
        axes = np.argmax(np.abs(normals), axis=1)
        normals *= np.where(normals[np.arange(len(core)), axes] < 0, -1., 1.)[:, None]
        valid = (count >= 5) & (values[:, 1] > (self.size*.25)**2)
        valid &= values[:, 0] < np.maximum(1e-9, values.sum(axis=1))*.085
        normals[~valid] = 0
        for key, normal, supported in zip(core, normals, valid):
            voxels[key]['normal'] = normal
            voxels[key]['normalValid'] = bool(supported)
            voxels[key]['normalPoint'] = voxels[key]['point'].copy()

    def rebuild(self, voxels, labels, dynamic_labels):
        self._refresh_normals(voxels)
        self.dirty.clear()
        self.revision += 1
        if not voxels:
            self.surfaces, self.volumes = {}, {}
            return
        data = self._arrays(voxels, labels, dynamic_labels)
        left, right, surface_edges, volume_edges = self._edges(data)
        surface_groups = component_groups(len(data['keys']), left[surface_edges], right[surface_edges])
        volume_groups = component_groups(len(data['keys']), left[volume_edges], right[volume_edges])
        self._build_volumes(volume_groups, data, voxels, labels)
        self._build_surfaces(surface_groups, data, voxels, labels)

    @staticmethod
    def _arrays(voxels, labels, dynamic_labels):
        keys = list(voxels)
        values = list(voxels.values())
        for voxel in values:
            if voxel['label'] and 'semanticNormal' not in voxel and np.linalg.norm(voxel.get('normal', 0)):
                voxel['semanticNormal'] = voxel['normal'].copy()
        return {'keys': keys, 'packed': pack_keys(keys), 'values': values,
                'points': np.asarray([v['point'] for v in values]),
                'colors': np.asarray([v['color'] for v in values]),
                'normals': np.asarray([v.get('normal', ZERO) for v in values]),
                'anchors': np.asarray([v.get('semanticNormal', ZERO) for v in values]),
                'instances': np.asarray([v.get('instanceId', '') for v in values]),
                'labels': np.asarray([v['label'] for v in values]),
                'dynamic': np.asarray([labels[v['label']] in dynamic_labels for v in values])}

    def _edges(self, data):
        order = np.argsort(data['packed'])
        packed = data['packed'][order]
        pairs = []
        # Half of the 26-neighborhood; diagonal observations remain connected.
        for x in (0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    if (x, y, z) <= (0, 0, 0):
                        continue
                    target = packed + x*(1 << 42) + y*(1 << 21) + z
                    indices = np.searchsorted(packed, target)
                    inside = indices < len(packed)
                    found = np.flatnonzero(inside)
                    found = found[packed[indices[found]] == target[found]]
                    pairs.append((order[found], order[indices[found]]))
        left = np.concatenate([pair[0] for pair in pairs])
        right = np.concatenate([pair[1] for pair in pairs])
        surface, volume = self._compatible(data, left, right)
        return left, right, surface, volume

    def _compatible(self, data, left, right):
        delta = data['points'][right]-data['points'][left]
        near = np.sum(delta*delta, axis=1) < (self.size*1.95)**2
        instances = data['instances']
        same = (instances[left] != '') & (instances[left] == instances[right])
        conflict = (instances[left] != '') & (instances[right] != '') & ~same
        label_conflict = (data['labels'][left] != 0) & (data['labels'][right] != 0)
        conflict |= label_conflict & (data['labels'][left] != data['labels'][right])
        conflict |= (data['dynamic'][left] | data['dynamic'][right]) & ~same
        a, b = data['normals'][left], data['normals'][right]
        planar = np.abs(np.sum(a*b, axis=1)) > .94
        planar &= (np.abs(np.sum(delta*a, axis=1)) < self.size*.55)
        planar &= (np.abs(np.sum(delta*b, axis=1)) < self.size*.55)
        # A corner can change a seed's estimated normal. Keep its semantic plane
        # as an anchor so a wall seed at the floor intersection cannot flood it.
        seed_left = (data['labels'][left] != 0) & (data['labels'][right] == 0)
        seed_right = (data['labels'][right] != 0) & (data['labels'][left] == 0)
        planar &= ~seed_left | (np.abs(np.sum(data['anchors'][left]*b, axis=1)) > .94)
        planar &= ~seed_right | (np.abs(np.sum(data['anchors'][right]*a, axis=1)) > .94)
        color = np.linalg.norm(data['colors'][left]-data['colors'][right], axis=1) < 65
        surface = near & ~conflict & planar & color
        sparse = (np.linalg.norm(a, axis=1) == 0) & (np.linalg.norm(b, axis=1) == 0)
        return surface, surface | (near & same) | (near & ~conflict & sparse & color)

    def _identity(self, prefix, group, values, previous, claimed):
        votes = Counter(values[i].get(prefix+'Id') for i in group)
        for identity, _ in votes.most_common():
            if identity in previous and identity not in claimed:
                claimed.add(identity)
                return identity
        self._serial += 1
        return f'{prefix}-{self._serial}'

    @staticmethod
    def _semantic(group, values):
        votes, weights = Counter(), Counter()
        for i in group:
            voxel = values[i]
            if voxel['label']:
                label = voxel['label']
                votes[label] += voxel.get('labelConfidence', 0.)
                weights[label] += 1
        if not votes:
            return 0, 0.
        label, score = votes.most_common(1)[0]
        return label, score/max(1., sum(weights.values()))

    def _build_volumes(self, groups, data, voxels, labels):
        previous, claimed, result = self.volumes, set(), {}
        assigned = np.zeros(len(data['keys']), bool)
        for value in data['values']:
            value.pop('surfaceColor', None)
        ranked = sorted((g for g in groups if len(g) >= 3),
                        key=lambda group: (-np.count_nonzero(data['labels'][group]), -len(group)))
        for group in ranked[:self.max_volumes]:
            if len(group) < 3:
                continue
            identity = self._identity('volume', group, data['values'], previous, claimed)
            semantic, confidence = self._semantic(group, data['values'])
            points = data['points'][group]
            instances = [data['values'][i].get('instanceId', '') for i in group]
            instance = Counter(filter(None, instances)).most_common(1)
            item = {'id': identity, 'label': labels[semantic], 'semantic': semantic,
                    'semanticConfidence': round(confidence, 3),
                    'instanceId': instance[0][0] if instance else None,
                    'centroid': points.mean(axis=0).round(3).tolist(),
                    'bounds': [(points.min(axis=0)-self.size/2).round(3).tolist(),
                               (points.max(axis=0)+self.size/2).round(3).tolist()],
                    'voxelCount': len(group), 'surfaceIds': [], 'geometry': 'observed-surface-volume',
                    'measuredRatio': round(np.mean([data['values'][i]['measured'] for i in group]), 3),
                    'lastSeen': max(data['values'][i]['lastSeen'] for i in group)}
            result[identity] = item
            self._inherit(group, data, item)
            assigned[group] = True
        for index, voxel in enumerate(voxels.values()):
            if not assigned[index]:
                voxel.pop('volumeId', None)
        self.volumes = result

    @staticmethod
    def _inherit(group, data, volume):
        # Connectivity already checked normals, colors and conflicting instances.
        sources = [data['values'][i] for i in group if data['values'][i]['label']]
        stamp = max((v.get('labelSeen', float('-inf')) for v in sources), default=float('-inf'))
        for i in group:
            voxel = data['values'][i]
            voxel['volumeId'] = volume['id']
            if not voxel['label'] and volume['semantic']:
                voxel.update(label=volume['semantic'], labelConfidence=volume['semanticConfidence'],
                             labelSeen=stamp, instanceId=volume['instanceId'] or '', inherited=True,
                             semanticNormal=voxel.get('normal', ZERO).copy())

    def _build_surfaces(self, groups, data, voxels, labels):
        previous, claimed, result, quads = self.surfaces, set(), {}, 0
        assigned = np.zeros(len(data['keys']), bool)
        ranked = sorted((g for g in groups if len(g) >= 5),
                        key=lambda group: (-np.count_nonzero(data['labels'][group]), -len(group)))
        for group in ranked:
            if len(result) >= self.max_surfaces or quads >= self.max_quads:
                break
            if len(group) < 5 or not np.linalg.norm(data['normals'][group[0]]):
                continue
            vertices, indices = self._mesh(group, data, self.max_quads-quads)
            if not indices:
                continue
            quads += len(indices)//6
            identity = self._identity('surface', group, data['values'], previous, claimed)
            item = self._surface(identity, group, data, labels, vertices, indices, previous)
            result[identity] = item
            assigned[group] = True
            for i in group:
                voxel = data['values'][i]
                voxel.update(surfaceId=identity, surfaceColor=item['color'])
            volume = self.volumes.get(item['volumeId'])
            if volume is not None:
                volume['surfaceIds'].append(identity)
        for index, voxel in enumerate(voxels.values()):
            if not assigned[index]:
                voxel.pop('surfaceId', None)
        self.surfaces = result

    def _surface(self, identity, group, data, labels, vertices, indices, previous):
        points = data['points'][group]
        measured = np.array([data['values'][i]['measured'] for i in group])
        color = np.median(data['colors'][group][measured] if measured.any() else data['colors'][group], axis=0)
        old = previous.get(identity)
        if old is not None:
            color = np.asarray(old['meanColor']) + np.clip(color-old['meanColor'], -4, 4)
        semantic, confidence = self._semantic(group, data['values'])
        return {'id': identity, 'volumeId': data['values'][group[0]].get('volumeId'),
                'normal': mean_normal(data['normals'][group]).round(4).tolist(),
                'centroid': points.mean(axis=0).round(3).tolist(),
                'bounds': [points.min(axis=0).round(3).tolist(), points.max(axis=0).round(3).tolist()],
                'vertices': vertices, 'indices': indices, 'color': packed_color(color),
                'meanColor': np.rint(color).astype(int).tolist(), 'semantic': semantic,
                'label': labels[semantic], 'confidence': round(confidence, 3),
                'measuredRatio': round(float(measured.mean()), 3), 'voxelCount': len(group),
                'continuous': True}

    def _mesh(self, group, data, budget):
        points, normals = data['points'][group], data['normals'][group]
        normal = mean_normal(normals)
        center = points.mean(axis=0)
        if np.percentile(np.abs((points-center)@normal), 95) < self.size*.55:
            return self._plane_tiles(group, data, normal, center, budget)
        # Curved components use local planes instead of one enclosing box.
        blocks = defaultdict(list)
        for i in group:
            blocks[tuple(v//self.block_width for v in data['keys'][i])].append(i)
        vertices, indices = [], []
        for local in blocks.values():
            if len(local) < 3 or budget <= 0:
                continue
            n = mean_normal(data['normals'][local])
            xyz, faces = self._plane_tiles(local, data, n, data['points'][local].mean(axis=0), budget)
            indices.extend(index+len(vertices)//3 for index in faces)
            vertices.extend(xyz)
            budget -= len(faces)//6
        return vertices, indices

    def _plane_tiles(self, group, data, normal, center, budget):
        axis = int(np.argmax(np.abs(normal)))
        tangent = [i for i in range(3) if i != axis]
        cells = {(data['keys'][i][tangent[0]], data['keys'][i][tangent[1]]) for i in group}
        tiles = np.asarray(list(rectangles(cells, budget)))
        if not len(tiles):
            return [], []
        corners = np.zeros((len(tiles), 4, 3))
        corners[:, :, tangent[0]] = tiles[:, [0, 2, 2, 0]]*self.size
        corners[:, :, tangent[1]] = tiles[:, [1, 1, 3, 3]]*self.size
        projected = (corners[:, :, tangent]-center[tangent])@normal[tangent]
        corners[:, :, axis] = center[axis]-projected/normal[axis]
        indices = np.arange(len(tiles))[:, None]*4 + [0, 1, 2, 0, 2, 3]
        return corners.round(4).ravel().tolist(), indices.ravel().tolist()


def rectangles(cells, budget):
    """Greedy occupied rectangles preserve every missing cell, including doorways."""
    remaining = set(cells)
    for x, y in sorted(cells, key=lambda cell: (cell[1], cell[0])):
        if (x, y) not in remaining or budget <= 0:
            continue
        right = x+1
        while (right, y) in remaining:
            right += 1
        top = y+1
        while all((column, top) in remaining for column in range(x, right)):
            top += 1
        remaining.difference_update((column, row) for column in range(x, right) for row in range(y, top))
        budget -= 1
        yield x, y, right, top

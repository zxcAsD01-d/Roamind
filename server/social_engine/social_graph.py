"""Measured proximity graph, not inferred willingness or calibrated probability."""
import math

import numpy as np

from .world_state import point3


def social_graph(people, self_position):
    located = {row['id']: row for row in people if point3(row.get('worldPosition')) is not None}
    links, adjacency = [], {key: set() for key in located}
    rows = list(located.values())
    for i, left in enumerate(rows):
        for right in rows[i+1:]:
            distance = float(np.linalg.norm((np.asarray(left['worldPosition'])-
                                             right['worldPosition'])[[0, 2]]))
            if distance > 2.4:
                continue
            score = math.exp(-distance/2)*min(left['confidence'], right['confidence'])
            links.append({'source': left['id'], 'target': right['id'],
                          'distance': round(distance, 3), 'proximityScore': round(score, 3)})
            adjacency[left['id']].add(right['id'])
            adjacency[right['id']].add(left['id'])
    groups = []
    remaining = set(located)
    while remaining:
        members = component(min(remaining), adjacency)
        remaining -= members
        if len(members) < 2:
            continue
        positions = np.asarray([located[key]['worldPosition'] for key in sorted(members)])
        center = positions.mean(axis=0)
        radius = max(.9, float(np.max(np.linalg.norm((positions-center)[:, [0, 2]], axis=1)))+.8)
        groups.append({'id': 'group:'+','.join(sorted(members)), 'members': sorted(members),
                       'center': center.tolist(), 'radius': round(radius, 3),
                       'entry': entry_point(center, radius, self_position),
                       'evidence': 'spatial-proximity-only', 'formation': 'unverified'})
    return {'edges': links, 'groups': groups, 'scoreMeaning': 'heuristic-not-probability'}


def component(start, adjacency):
    found, pending = set(), [start]
    while pending:
        key = pending.pop()
        if key in found:
            continue
        found.add(key)
        pending.extend(adjacency[key]-found)
    return found


def entry_point(center, radius, self_position):
    origin = point3(self_position)
    if origin is None:
        return None
    direction = origin-center
    direction[1] = 0
    length = float(np.linalg.norm(direction))
    if length < .01:
        return None
    return (center+direction/length*radius).tolist()

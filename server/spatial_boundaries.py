"""Metric floor boundary segments; missing evidence never becomes a wall."""
from collections import defaultdict


BOUNDARY_SECONDS = 120


def boundary_state(cell, now):
    if cell is None or now-cell[1] >= 1800 or cell[0] <= .4:
        return 'unknown'
    if cell[0] > 1.5 and cell[2] >= 3 and now-cell[1] < BOUNDARY_SECONDS:
        return 'supported'
    return 'uncertain'


def boundary_geometry(key, other):
    x, z = key
    if other[0] != x:
        return ('x', max(x, other[0]), z)
    return ('z', max(z, other[1]), x)


def boundary_runs(lines, resolution):
    segments = []
    for (axis, line, state, target), offsets in sorted(lines.items()):
        ordered = sorted(offsets)
        start = previous = ordered[0]
        for offset in [*ordered[1:], ordered[-1]+2]:
            if offset != previous+1:
                ends = [[line, 0., start], [line, 0., previous+1]] if axis == 'x' else [
                    [start, 0., line], [previous+1, 0., line]]
                segments.append({'start': [round(v*resolution, 4) for v in ends[0]],
                                 'end': [round(v*resolution, 4) for v in ends[1]],
                                 'state': state, 'neighborRegion': target or None})
                start = offset
            previous = offset
    return segments


def describe_boundaries(assignments, cells, resolution, now, neighbors):
    lines, counts = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(int))
    for key, identity in assignments.items():
        for other in neighbors(key):
            target = assignments.get(other)
            if target == identity:
                continue
            state = 'opening' if target is not None else boundary_state(cells.get(other), now)
            axis, line, offset = boundary_geometry(key, other)
            lines[identity][(axis, line, state, target or 0)].add(offset)
            counts[identity][state] += 1
    return {identity: {'counts': dict(counts[identity]),
                       'segments': boundary_runs(boundary, resolution)}
            for identity, boundary in lines.items()}


def describe_spaces(regions, exits, resolution, now):
    grouped = defaultdict(list)
    for region in regions.values():
        grouped[region['componentId']].append(region)
    result = []
    for identity, group in sorted(grouped.items()):
        members = {region['id'] for region in group}
        counts = defaultdict(int)
        for region in group:
            for segment in region['boundarySegments']:
                if segment['neighborRegion'] in members:
                    continue
                length = sum(abs(a-b) for a, b in zip(segment['start'], segment['end']))
                counts[segment['state']] += round(length/resolution)
        outward = [item for item in exits if members.intersection(item['regions']) and not item['connectsRegions']]
        ambiguous = any(item['normal'] is not None and item['state'] != 'blocked' for item in outward)
        total = sum(counts.values())
        closed = total > 0 and counts['supported'] == total and not ambiguous
        status = 'observed-enclosed' if closed else 'open-unobserved' if counts['unknown'] else 'uncertain-boundary'
        result.append({'id': identity, 'regionIds': sorted(members),
                       'area': round(sum(region['area'] for region in group), 2),
                       'boundaryEvidence': dict(counts), 'exitIds': [item['id'] for item in outward],
                       'enclosureConfidence': round(counts['supported']/max(1, total), 3),
                       'enclosure': {'status': status, 'closedBoundary': closed,
                                     'scope': 'observed-floor-boundary', 'observedAt': now}})
    return result

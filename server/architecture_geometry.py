"""Exterior window openings and frames; the same solids render and collide."""


def window_wall(z, openings, name):
    solids, cursor = [], -10.
    def solid(label, x, y, width, height, material='wall', depth=.16):
        return {'id': name+':'+label, 'center': [x, y, z], 'size': [width, height, depth],
                'material': material, 'kind': 'wall'}
    for i, (center, width) in enumerate(openings):
        left, right = center-width/2, center+width/2
        if left > cursor:
            solids.append(solid(f'pier-{i}', (cursor+left)/2, 1.4, left-cursor, 2.8))
        solids += [solid(f'base-{i}', center, .45, width, .9),
                   solid(f'lintel-{i}', center, 2.5, width, .6),
                   solid(f'glass-{i}', center, 1.55, width, 1.3, 'glass', .025)]
        for side in (-1, 1):
            solids.append(solid(f'jamb-{i}-{side}', center+side*(width/2-.035), 1.55, .07, 1.3, 'oak', .2))
            solids.append(solid(f'rail-{i}-{side}', center, 1.55+side*.63, width, .06, 'oak', .2))
        solids.append(solid(f'mullion-{i}', center, 1.55, .04, 1.3, 'metal', .08))
        cursor = right
    if cursor < 10:
        solids.append(solid('end', (cursor+10)/2, 1.4, 10-cursor, 2.8))
    return solids

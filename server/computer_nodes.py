"""Join observed input surfaces and observed seats; never query simulation layout."""
import copy
import math


def workstation(row, objects, stature, now):
    if objects is None:
        return None
    for keyboard in row.nodes:
        if keyboard['kind'] != 'workstation':
            continue
        candidates = []
        for chair in objects.values():
            if (chair.identity == row.identity or chair.confidence < .4
                    or not -.1 <= now-chair.observed < 1.5):
                continue
            for seat in chair.nodes:
                if seat['kind'] != 'support' or min(seat['size'][0], seat['size'][2]) < .28:
                    continue
                horizontal = math.hypot(*(keyboard['point'][i]-seat['point'][i] for i in (0, 2)))
                vertical = keyboard['point'][1]-seat['point'][1]
                if .32 <= horizontal <= stature*.8 and .08 <= vertical <= stature*.35:
                    candidates.append((horizontal, chair.identity, seat, chair.observed))
        if candidates:
            _, identity, seat, stamp = min(candidates, key=lambda item: (item[0], item[1]))
            return {**copy.deepcopy(keyboard), 'seatObjectId': identity,
                    'seat': {**copy.deepcopy(seat), 'observedAt': stamp}}
    return None


def approach(memory, target, distance):
    seat, keyboard = target['seat']['point'], target['point']
    dx, dz = keyboard[0]-seat[0], keyboard[2]-seat[2]
    length = math.hypot(dx, dz)
    if length < .1:
        return None
    dx, dz = dx/length, dz/length
    paths, candidates = memory.reachable(), []
    for offset in (0, -.08, .08):
        point = [seat[0]+dx*distance-dz*offset, float(memory.position[1]), seat[2]+dz*distance+dx*offset]
        key = memory.key(point)
        if key in paths.parents and memory.cells.get(key, (0,))[0] < -.15:
            candidates.append((paths.lengths[key]+abs(offset), point))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def facing_seat(position, target):
    seat, keyboard = target['seat']['point'], target['point']
    a = [position[i]-seat[i] for i in (0, 2)]
    b = [keyboard[i]-seat[i] for i in (0, 2)]
    norm = math.hypot(*a)*math.hypot(*b)
    return norm > .01 and sum(x*y for x, y in zip(a, b))/norm >= .9

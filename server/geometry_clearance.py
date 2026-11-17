"""Exact XZ circle/rectangle and swept-circle clearance, without square corners."""
import math


def box_distance_squared(x, z, center, size):
    return max(0., abs(x-center[0])-size[0]/2)**2 + max(0., abs(z-center[2])-size[2]/2)**2


def point_segment_squared(p, a, b):
    delta = (b[0]-a[0], b[1]-a[1])
    length = delta[0]**2+delta[1]**2
    t = max(0., min(1., ((p[0]-a[0])*delta[0]+(p[1]-a[1])*delta[1])/length)) if length else 0.
    return (p[0]-a[0]-delta[0]*t)**2+(p[1]-a[1]-delta[1]*t)**2


def segment_box_squared(a, b, center, half):
    enter, leave = 0., 1.
    for axis in range(2):
        delta = b[axis]-a[axis]
        low, high = center[axis]-half[axis], center[axis]+half[axis]
        if abs(delta) < 1e-12:
            if not low <= a[axis] <= high:
                enter = math.inf
                break
        else:
            first, last = sorted(((low-a[axis])/delta, (high-a[axis])/delta))
            enter, leave = max(enter, first), min(leave, last)
    if enter <= leave:
        return 0.
    distances = [sum(max(0., abs(p[i]-center[i])-half[i])**2 for i in range(2)) for p in (a, b)]
    distances.extend(point_segment_squared((center[0]+x*half[0], center[1]+z*half[1]), a, b)
                     for x in (-1, 1) for z in (-1, 1))
    return min(distances)

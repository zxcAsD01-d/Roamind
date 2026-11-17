"""Navigation-only motion association; never authorizes body contact."""
import numpy as np


def horizontal(person):
    point = person.get('worldPosition')
    if not isinstance(point, list) or len(point) != 3 or not np.isfinite(point).all():
        return None
    return np.asarray(point)[[0, 2]]


def associate_person(candidates, previous, stamp):
    if not previous:
        return min(candidates, key=lambda person: person['distance'], default=None)
    direct = next((person for person in candidates if person['id'] in
                   (previous['id'], previous.get('observationId'))), None)
    origin = horizontal(previous)
    elapsed = stamp-previous['observedAt']
    if elapsed > 2.5:
        return min(candidates, key=lambda person: person['distance'], default=None)
    if origin is None:
        return direct or min(candidates, key=lambda person: person['distance'], default=None)
    if elapsed < 0:
        return direct
    if elapsed == 0:
        if direct is not None:
            direct.update(observationId=direct['id'], id=previous['id'],
                          navigationVelocity=previous.get('navigationVelocity', [0., 0.]))
        return direct
    velocity = np.asarray(previous.get('navigationVelocity', [0., 0.]))
    prediction = origin+velocity*elapsed
    plausible = []
    for person in candidates:
        point = horizontal(person)
        if point is not None and np.linalg.norm(point-origin) <= .25+1.6*elapsed:
            plausible.append((float(np.linalg.norm(point-prediction)), person))
    plausible.sort(key=lambda item: item[0])
    if direct is None:
        if not plausible or (len(plausible) > 1 and plausible[1][0]-plausible[0][0] < .3):
            return None
        direct = plausible[0][1]
    point = horizontal(direct)
    if point is not None:
        measured = (point-origin)/elapsed
        speed = np.linalg.norm(measured)
        measured *= min(1., 1.6/max(speed, 1e-6))
        direct.update(observationId=direct['id'], id=previous['id'],
                      navigationVelocity=(.5*velocity+.5*measured).tolist(),
                      association='unique-visual-motion-navigation-only')
    return direct

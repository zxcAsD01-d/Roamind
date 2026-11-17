"""Selected contact chain over a stopped stance; root remains unchanged."""
import numpy as np
from scipy.spatial.transform import Rotation


def forward(local, root, rest, parents):
    world = Rotation.from_rotvec(local.reshape(27, 3)).as_matrix()
    posed = np.empty((27, 3)); posed[0] = root
    for joint, parent in enumerate(parents[1:], 1):
        posed[joint] = posed[parent]+world[parent]@(rest[joint]-rest[parent])
        world[joint] = world[parent]@world[joint]
    return posed, world


def align(a, b):
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    axis = np.cross(a, b); sine = np.linalg.norm(axis); cosine = np.clip(a@b, -1, 1)
    if sine < 1e-7:
        if cosine > 0:
            return np.eye(3)
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))]); axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis*np.pi).as_matrix()
    return Rotation.from_rotvec(axis/sine*np.arctan2(sine, cosine)).as_matrix()


def solve_frame(local, root, target, rest, parents, hand, previous_elbow):
    upper, lower = parents[parents[hand]], parents[hand]
    posed, world = forward(local, root, rest, parents)
    shoulder, elbow, wrist = posed[upper], posed[lower], posed[hand]
    first, second = np.linalg.norm(elbow-shoulder), np.linalg.norm(wrist-elbow)
    delta = target-shoulder; distance = np.linalg.norm(delta)
    if not abs(first-second) < distance <= first+second+.002:
        raise ValueError('生成姿态下触摸目标不可达；请重新调整站位')
    direction = delta/distance
    distance = min(distance, first+second-.00001)
    target = shoulder+direction*distance
    bend = (elbow if previous_elbow is None else previous_elbow)-shoulder
    bend -= direction*(bend@direction)
    if np.linalg.norm(bend) < .001:
        bend = np.cross(direction, [0., 1., 0.])
    if np.linalg.norm(bend) < .001:
        bend = np.cross(direction, [1., 0., 0.])
    along = (first*first-second*second+distance*distance)/(2*distance)
    desired = shoulder+direction*along+bend/np.linalg.norm(bend)*np.sqrt(max(0, first*first-along*along))
    next_world = align(elbow-shoulder, desired-shoulder)@world[upper]
    local[upper*3:upper*3+3] = Rotation.from_matrix(world[parents[upper]].T@next_world).as_rotvec()
    posed, world = forward(local, root, rest, parents)
    next_world = align(posed[hand]-posed[lower], target-posed[lower])@world[lower]
    local[lower*3:lower*3+3] = Rotation.from_matrix(world[upper].T@next_world).as_rotvec()
    return desired


def refine(window, origin, plan, frame, previous_local, previous_root, rest, parents):
    from contact_model import check_support
    descriptor = plan['effector']
    chain, end = descriptor['chain'], descriptor['joint']
    arm = chain+[end]
    shift = origin-np.array([0, .92, 0])
    previous = previous_local[-1].copy()
    elbow = forward(previous, previous_root[-1]-shift, rest, parents)[0][chain[-1]]
    for index in range(len(window.root)):
        local = window.local[index]
        # Preserve every channel outside the explicitly selected contact chain.
        stance = np.asarray(plan['stanceLocal'])
        for joint in range(27):
            if joint not in arm:
                local[joint*3:joint*3+3] = stance[joint*3:joint*3+3]
        local[end*3:end*3+3] = stance[end*3:end*3+3]
        for joint in chain:
            local[joint*3:joint*3+3] = previous[joint*3:joint*3+3]
        shoulder = forward(local, window.root[index]-shift, rest, parents)[0][arm[0]]
        target = np.asarray(plan['handTrack'][frame+index])+shoulder-plan['sourceShoulder']
        if descriptor['kind'] == 'axial':
            solve_axial(local, window.root[index]-shift, target, rest, parents, descriptor, stance)
        else:
            elbow = solve_frame(local, window.root[index]-shift, target, rest, parents, end, elbow)
        for joint in chain:
            local[joint*3:joint*3+3] = bounded_rotation(previous[joint*3:joint*3+3], local[joint*3:joint*3+3], .25)
        posed = forward(local, window.root[index]-shift, rest, parents)[0]
        check_support(posed, descriptor)
        elbow = posed[chain[-1]]
        previous = local.copy()
    return window


def solve_axial(local, root, target, rest, parents, descriptor, stance):
    """Damped position Jacobian, bounded chest/neck rotation relative to the stance."""
    chain, end = descriptor['chain'], descriptor['joint']
    for _ in range(18):
        posed, world = forward(local, root, rest, parents)
        error = target-posed[end]
        if np.linalg.norm(error) < .012:
            return
        axes = [world[parents[joint]][:, axis] for joint in chain for axis in range(3)]
        jacobian = np.column_stack([np.cross(axis, posed[end]-posed[joint])
                                   for axis, joint in zip(axes, np.repeat(chain, 3))])
        delta = jacobian.T@np.linalg.solve(jacobian@jacobian.T+np.eye(3)*.002, error)
        for k, joint in enumerate(chain):
            before = Rotation.from_rotvec(local[joint*3:joint*3+3])
            rotation = (Rotation.from_rotvec(np.clip(delta[k*3:k*3+3], -.06, .06))*before).as_rotvec()
            limit = .55 if joint == 5 else .22
            local[joint*3:joint*3+3] = bounded_rotation(stance[joint*3:joint*3+3], rotation, limit)


def bounded_rotation(previous, proposed, limit):
    before, after = Rotation.from_rotvec(previous), Rotation.from_rotvec(proposed)
    delta = (before.inv()*after).as_rotvec()
    angle = np.linalg.norm(delta)
    return (before*Rotation.from_rotvec(delta*min(1., limit/max(angle, 1e-8)))).as_rotvec()

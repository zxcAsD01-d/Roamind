"""Optimize candidate clips with measured quality gates; retain originals on regression."""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic_avatar import AvatarCalibration
from supermimic_bridge import adapt_human, reverse_human, track_adapted
from supermimic_policy import SuperMimicPolicy
from supermimic_refinement import residual_candidates
from supermimic_skeleton import HumanMotion


def foot_measurement(motion: HumanMotion) -> tuple[np.ndarray, np.ndarray]:
    indices = [motion.semantic[name] for name in ('leftFoot', 'leftToes', 'rightFoot', 'rightToes') if name in motion.semantic]
    feet = motion.positions[:, indices] + motion.root[:, None]
    fps = motion.document['fps']
    floor = np.percentile(feet[:, :, 1], 20, axis=0, method='lower')
    support = np.maximum(feet[:-1, :, 1], feet[1:, :, 1]) < floor + .012
    support &= np.abs(np.diff(feet[:, :, 1], axis=0)) * fps < .08
    return feet, support


def quality(document: dict, reference: dict | None = None) -> dict:
    motion = HumanMotion.load(document)
    feet, support = foot_measurement(motion)
    speed = np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=2) * document['fps']
    local = Rotation.from_rotvec(motion.local.reshape(-1, 3)).as_matrix().reshape(len(motion.root), -1, 3, 3)
    increments = Rotation.from_matrix((local[1:] @ local[:-1].transpose(0, 1, 3, 2)).reshape(-1, 3, 3)).as_rotvec()
    angular = np.linalg.norm(increments, axis=1)
    slip = speed[support]
    report = {'supportSamples': int(support.sum()), 'supportMean': float(slip.mean()) if len(slip) else None,
            'supportP95': float(np.percentile(slip, 95, method='lower')) if len(slip) else None,
            'angularStepMax': float(angular.max()), 'rootHeightRange': float(np.ptp(motion.root[:, 1])),
            'footHeightMin': float(feet[:, :, 1].min()), 'footHeightRange': float(np.ptp(feet[:, :, 1]))}
    if reference is not None:
        _, original_support = foot_measurement(HumanMotion.load(reference))
        paired = speed[original_support]
        report.update(pairedSupportMean=float(paired.mean()) if len(paired) else None,
                      pairedSupportP95=float(np.percentile(paired, 95, method='lower')) if len(paired) else None)
    return report


def acceptance(before: dict, after: dict, root_error: float, reconstruction: dict) -> list[str]:
    reasons = []
    if min(before['supportSamples'], after['supportSamples']) < 4:
        return ['insufficient measured support samples']
    if after['supportSamples'] < before['supportSamples'] * .6:
        reasons.append('support coverage reduced')
    if after['supportMean'] > min(.12, before['supportMean'] * .95):
        reasons.append('mean support slip did not improve by 5%')
    if after['supportP95'] > min(.32, before['supportP95'] + .005):
        reasons.append('support P95 regressed')
    if after['angularStepMax'] > min(.4, before['angularStepMax'] + .025):
        reasons.append('joint continuity regressed')
    if after.get('pairedSupportMean', 0.) > before['supportMean']:
        reasons.append('slip on original support frames regressed')
    if after.get('pairedSupportP95', 0.) > before['supportP95'] + .005:
        reasons.append('P95 on original support frames regressed')
    if root_error > .12:
        reasons.append('root trajectory error exceeds 12cm')
    if reconstruction['footTargetMaxErrorM'] > .025:
        reasons.append('human leg reach error exceeds 2.5cm')
    return reasons


def select_correction(human: HumanMotion, physical: list[dict], before: dict) -> tuple[dict, dict]:
    evaluated = []
    for raw in physical:
        for document, settings in residual_candidates(human, raw):
            after = quality(document, human.document)
            difference = np.array(document['rootDelta']).reshape(-1, 3) - human.root
            root_error = float(np.linalg.norm(difference, axis=1).max())
            reconstruction = document['supermimicReconstruction']
            reasons = acceptance(before, after, root_error, reconstruction)
            score = (after['supportMean'] + .1 * after['supportP95']) if after['supportMean'] is not None else float('inf')
            evaluated.append((bool(reasons), score, document, {'after': after, 'rootErrorMaxM': root_error,
                'reconstruction': reconstruction, 'refinement': settings, 'accepted': not reasons,
                'rejectionReasons': reasons}))
    _, _, document, report = min(evaluated, key=lambda item: item[:2])
    report['evaluatedCandidates'] = len(evaluated)
    return document, report


def boundary_change(source: HumanMotion, document: dict) -> dict:
    corrected = HumanMotion.load(document)
    a = Rotation.from_rotvec(source.local.reshape(-1, 3))
    b = Rotation.from_rotvec(corrected.local.reshape(-1, 3))
    changes = np.linalg.norm((b * a.inv()).as_rotvec().reshape(source.local.shape), axis=2)
    root = np.linalg.norm(corrected.root - source.root, axis=1)
    return {'entryPoseMaxRad': float(changes[0].max()), 'exitPoseMaxRad': float(changes[-1].max()),
            'entryRootM': float(root[0]), 'exitRootM': float(root[-1]),
            'requiresTransitionReindex': True}


def optimize_motion(document: dict, policy: SuperMimicPolicy, avatar: dict | None = None,
                    root_scale: float | None = None) -> dict:
    started = time.perf_counter()
    human = HumanMotion.load(document)
    calibration = AvatarCalibration.load(human, avatar, root_scale) if avatar is not None else None
    if calibration:
        human = calibration.human
    adapted = adapt_human(human, policy)
    tracking, trajectory = track_adapted(adapted, policy)
    report = {'accepted': False, 'tracking': tracking, 'scaleHumanToG1': adapted.skeleton.scale,
              'fitRmsMeanRad': float(np.mean(adapted.fit_errors)),
              'fitP95Ms': float(np.percentile(adapted.fit_ms, 95)), 'before': quality(human.document),
              'backend': policy.backend, 'method': 'G1 physics tracking; fixed pretrained weights',
              'avatarValidated': False, 'elapsedMs': 0.}
    if calibration:
        report.update(avatar=calibration.name, avatarRootScale=calibration.root_scale,
                      qualitySpace='measured avatar geometry')
    candidate = None
    if tracking['completed']:
        physical = [reverse_human(adapted, trajectory, foot_mode=mode) for mode in ('residual', 'trajectory')]
        candidate, selected = select_correction(human, physical, report['before'])
        report.update(selected)
        report['boundary'] = boundary_change(human, candidate)
        ref = adapted.reference
        projection = reverse_human(adapted, {'root': ref.position, 'rotation': ref.rotation, 'joints': ref.joints},
                                   foot_mode='trajectory')
        _, ablation = select_correction(human, [projection], report['before'])
        report['withoutPolicy'] = ablation
        if calibration:
            candidate = calibration.restore(candidate)
    else:
        report['rejectionReasons'] = ['G1 physical tracking terminated']
    report['elapsedMs'] = (time.perf_counter() - started) * 1000
    # Candidates remain available for explicit research comparison. Rejected clips never replace input.
    return {'report': report, 'motion': candidate if report['accepted'] else document, 'candidate': candidate}

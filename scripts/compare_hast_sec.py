"""Run unmodified HAST-SEC modules against local clips without changing playback."""
from __future__ import annotations

import hashlib
import itertools
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from library_motion import blend_start, joints  # noqa: E402

FPS = 20
OUT = ROOT / 'output/interaction'


def read_clips() -> list[dict]:
    database = ROOT / '.local-data/motion-library.sqlite3'
    with sqlite3.connect(f'{database.as_uri()}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT id,action,motion FROM clips ORDER BY id').fetchall()
    clips = [{**json.loads(raw), 'id': identity, 'group': 'library', 'action': action}
             for identity, action, raw in rows]
    bank = json.loads((ROOT / 'public/locomotion/bank.json').read_text())
    return clips + [{**clip, 'id': clip['name'], 'group': 'preset', 'action': clip['kind']}
                    for clip in bank['clips']]


def channels(clip: dict) -> tuple[np.ndarray, np.ndarray]:
    frames = clip['frames']
    local = np.asarray(clip['localRot']).reshape(frames, 27, 3)
    root = np.asarray(clip['rootDelta']).reshape(frames, 3)
    if not np.isfinite(local).all() or not np.isfinite(root).all():
        raise ValueError('测试素材存在非有限数值')
    return local, root


def baseline(old: dict, local: np.ndarray, root: np.ndarray) -> dict:
    local, root = local.copy(), root.copy()
    blend_start(local, old)
    height_offset = np.asarray(old['rootDelta'])[-2] - root[0, 1]
    count = min(8, len(root))
    t = np.arange(1, count + 1) / 8
    root[:count, 1] += height_offset * (1 - t**3 * (10 + t * (-15 + 6*t)))
    return {'local': local.tolist(), 'root': root.tolist()}


def make_case(old: dict, incoming: dict) -> dict:
    before, before_root = channels(old)
    local, root = channels(incoming)
    count = min(80, len(before), len(local))
    local, root = local[:count].copy(), root[:count].copy()
    root[:, [0, 2]] += before_root[-1, [0, 2]] - root[0, [0, 2]]
    return {'id': old['id'] + ' -> ' + incoming['id'], 'group': old['group'],
            'oldAction': old['action'], 'newAction': incoming['action'],
            'loop': old['id'] == incoming['id'], 'bones': incoming['bones'],
            'parents': incoming['parents'], 'restJoints': incoming['restJoints'],
            'oldLocal': before[-count:].tolist(), 'oldRoot': before_root[-count:].tolist(),
            'local': local.tolist(), 'root': root.tolist(),
            'current': baseline(old, local, root)}


def make_cases(clips: list[dict]) -> list[dict]:
    stored = [clip for clip in clips if clip['group'] == 'library']
    presets = [clip for clip in clips if clip['group'] == 'preset']
    pairs = list(itertools.product(stored, repeat=2)) + [(clip, clip) for clip in presets]
    for first, second in [('idle', 'start'), ('start', 'walk'), ('walk', 'stop'), ('walk', 'turn')]:
        a = [clip for clip in presets if clip['action'] == first]
        b = [clip for clip in presets if clip['action'] == second]
        pairs.extend(zip(a, b))
    return [make_case(old, new) for old, new in pairs]


def angular_steps(local: np.ndarray) -> np.ndarray:
    q = Rotation.from_rotvec(local.reshape(-1, 3)).as_quat().reshape(-1, 27, 4)
    dot = np.abs(np.sum(q[1:] * q[:-1], axis=-1))
    return 2 * np.arccos(np.clip(dot, 0, 1))


def pose_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa = Rotation.from_rotvec(a.reshape(-1, 3))
    qb = Rotation.from_rotvec(b.reshape(-1, 3))
    return (qa.inv() * qb).magnitude().reshape(-1, 27)


def feet(case: dict, local: np.ndarray, root: np.ndarray) -> np.ndarray:
    motion = {**case, 'frames': len(root), 'fps': FPS, 'up': 'yup',
              'localRot': local.ravel().tolist(), 'rootDelta': root.ravel().tolist()}
    indices = [case['bones'].index(name) for name in ['leftFoot', 'leftToes', 'rightFoot', 'rightToes']]
    return joints(motion)[:, indices]


def support_metrics(case: dict, local: np.ndarray, root: np.ndarray) -> dict:
    reference = feet(case, np.asarray(case['local']), np.asarray(case['root']))
    actual = feet(case, local, root)
    floor = np.percentile(reference[:, :, 1], 15, axis=0)
    mask = np.maximum(reference[:-1, :, 1], reference[1:, :, 1]) < floor + .025
    mask &= np.abs(np.diff(reference[:, :, 1], axis=0)) * FPS < .12
    # Fixed source-derived mask: smoothing cannot hide slip by changing contact labels.
    speed = np.linalg.norm(np.diff(actual[:, :, [0, 2]], axis=0), axis=2) * FPS
    slip = speed[mask]
    return {'supportSamples': int(mask.sum()), 'supportMeanMps': float(slip.mean()) if len(slip) else None,
            'supportP95Mps': float(np.percentile(slip, 95)) if len(slip) else None,
            'footHeightMinM': float(actual[:, :, 1].min())}


def continuity_metrics(case: dict, result: dict) -> dict:
    local, root = np.asarray(result['local']), np.asarray(result['root'])
    reference, target_root = np.asarray(case['local']), np.asarray(case['root'])
    history = np.asarray(result.get('oldLocal', case['oldLocal']))
    history_root = np.asarray(result.get('oldRoot', case['oldRoot']))
    combined = np.concatenate((history[-2:], local))
    steps = angular_steps(combined)
    q = Rotation.from_rotvec(combined.reshape(-1, 3)).as_quat().reshape(-1, 27, 4)
    velocity = np.stack([(Rotation.from_quat(a).inv() * Rotation.from_quat(b)).as_rotvec()
                         for a, b in zip(q[:-1], q[1:])]) * FPS
    root_steps = np.linalg.norm(np.diff(np.concatenate((history_root[-1:], root)), axis=0), axis=1)
    lag_errors = [float(pose_error(local[lag:], reference[:len(local)-lag]).mean())
                  for lag in range(min(9, len(local)//2))]
    return {'angularStepMaxRad': float(steps[1:].max()), 'seamPoseMaxRad': float(steps[1].max()),
            'seamVelocityJumpMaxRadps': float(np.linalg.norm(velocity[1]-velocity[0], axis=1).max()),
            'postAnchorVelocityJumpMaxRadps': float(np.linalg.norm(velocity[2]-velocity[0], axis=1).max()),
            'rootStepMaxM': float(root_steps.max()),
            'rootDeviationMaxM': float(np.linalg.norm(root-target_root, axis=1).max()),
            'targetPoseErrorMeanRad': float(pose_error(local, reference).mean()),
            'angularTravelRatio': float(angular_steps(local).sum()/max(1e-9, angular_steps(reference).sum())),
            'trackingLagEstimateMs': int(np.argmin(lag_errors))*50,
            **support_metrics(case, local, root)}


def summarize(cases: list[dict]) -> dict:
    groups = {'all': cases, 'loops': [c for c in cases if c['loop']],
              'moving': [c for c in cases if c['moving']]}
    result = {}
    for name, group in groups.items():
        methods = {}
        for method in group[0]['methods']:
            rows = [c['methods'][method] for c in group]
            fields = ['angularStepMaxRad', 'seamPoseMaxRad', 'seamVelocityJumpMaxRadps',
                      'rootDeviationMaxM', 'targetPoseErrorMeanRad', 'trackingLagEstimateMs',
                      'supportMeanMps', 'supportP95Mps', 'angularTravelRatio']
            methods[method] = {field + 'Median': float(np.median([r[field] for r in rows if r[field] is not None]))
                               for field in fields}
            methods[method]['angularStepWorstRad'] = max(r['angularStepMaxRad'] for r in rows)
        result[name] = {'count': len(group), 'methods': methods}
    return result


def evaluate(fixtures: dict, runtime: dict) -> list[dict]:
    cases = []
    for case, samples in zip(fixtures['cases'], runtime['cases'], strict=True):
        if case['id'] != samples['id']:
            raise ValueError('测试输出与输入案例顺序不一致')
        methods = {'raw': {'local': case['local'], 'root': case['root']},
                   'current': case['current'], **samples['methods']}
        root = np.asarray(case['root'])
        cases.append({'id': case['id'], 'loop': case['loop'], 'group': case['group'],
                      'moving': bool(np.linalg.norm(np.ptp(root[:, [0, 2]], axis=0)) > .15),
                      'methods': {name: continuity_metrics(case, value) for name, value in methods.items()}})
    return cases


def write_previews(fixtures: dict, runtime: dict) -> None:
    selectors = [('wave', 'raise', False), ('point', 'wave', False), ('wave', 'wave', True),
                 ('squat', 'crouch', False), ('walk', 'walk', True), ('walk', 'turn', False),
                 ('idle', 'start', False), ('turn', 'turn', True)]
    index = []
    for old, new, loop in selectors:
        match = next((i for i, case in enumerate(fixtures['cases'])
                      if case['oldAction'] == old and case['newAction'] == new and case['loop'] == loop), None)
        if match is None:
            continue
        fixture, samples = fixtures['cases'][match], runtime['cases'][match]
        file = f'hast-sec-preview-{len(index)}.json'
        (OUT/file).write_text(json.dumps({'fixture': fixture, 'samples': samples}, separators=(',', ':')))
        index.append({'id': fixture['id'], 'label': f'{old} → {new}' + ('（循环接缝）' if loop else ''), 'file': file})
    (OUT/'hast-sec-preview-index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')


def write_report(report: dict) -> None:
    native = report['nativeLibrary']
    lines = ['# HAST-SEC 复用对照实验', '', f"上游版本：`{report['revision']}`。", '',
             '结论：动作层适合继续试验局部手臂动作；不能直接替换整身移动管线。', '',
             f"原生库：{native['classes']} 类、{native['variants']} 变体、{native['sources']} 个来源，"
             f"每个变体循环 4 次，共采样 {native['frames']} 帧；固定输入重复回放一致，未蒙版关节保持原样。",
             f"中断检查 {native['interruptionCases']} 组；最大单帧转角 {native['interruptionMaxStepRad']:.3f} rad。", '',
             '| 对照方法 | 单帧最大关节角的案例中位数 rad | 移动案例脚速 P95 的中位数 m/s | 延迟估计中位数 ms |',
             '|---|---:|---:|---:|']
    names = {'current': '现有特殊动作八帧接缝', 'hastResidual': 'HAST 位姿与速度残差',
             'hastLayers': 'HAST 动作层淡入', 'hastDamped': 'HAST 二阶阻尼',
             'hastDampedPreserveRoot': 'HAST 二阶阻尼，保留根运动'}
    for key, name in names.items():
        a = report['summary']['all']['methods'][key]
        m = report['summary']['moving']['methods'][key]
        lines.append(f"| {name} | {a['angularStepMaxRadMedian']:.3f} | {m['supportP95MpsMedian']:.3f} | {a['trackingLagEstimateMsMedian']:.0f} |")
    continuity = native['continuity']
    lines.extend(['', f"共 {report['summary']['all']['count']} 案例；移动案例 {report['summary']['moving']['count']} 组。", '',
                  f"原生底座连续 {continuity['seconds']} 秒、切换 {continuity['switches']} 次；"
                  f"旧过渡边界最大 {continuity['baselineBoundaryMaxRad']:.3f} rad，"
                  f"残差版 {continuity['residualBoundaryMaxRad']:.3f} rad，最坏关节为 {continuity['worstBoundary']['bone']}。",
                  f"原生动作层全程最大单帧角步长 {native['maxStepRad']:.3f} rad，循环边界最大 {native['cycleBoundaryMaxRad']:.3f} rad。", '',
                  '限制：这是原始模块的离线实验，不是整套 HAST 全身可行性求解器测试。'
                  '当前 27 个独立旋转通道补齐到上游 53 通道契约，没有完成 HAST 原生骨架到当前骨架的重定向。',
                  '基础预设也套用同一特殊动作接缝基线，只用于算法对照；不代表主页面经过相位选择、步幅校准的预设播放质量。',
                  '移动片段只对齐 XZ 起点，不强改路线或支撑相位。支撑掩码从原始片段几何计算后固定；没有支撑样本时记录 null。',
                  'HAST 动作层没有根移动轨迹；旋转更平滑并不代表落脚更稳定。阻尼的延迟为 20 Hz 网格搜索估计，精度 50 ms。',
                  'Wolferia 对照页保留原重定向，没有脚底 IK、锁脚或骨盆补偿。整段循环预览会显式重置，不能据此声称累计漂移为零。', '',
                  '复现：`npm run test:hast`。开发服务对照页：`/hast-comparison.html`。',
                  '上游自检：123 个模块、62 个资源通过。当前行为、动作库、重叠、连续性和预设共 35 项测试通过，构建通过。',
                  '上游 LICENSE 明确为非商业学习研究许可；完整条款与第三方资源声明保留于 external/HAST-SEC。', ''])
    (OUT/'hast-sec-report.md').write_text('\n'.join(lines))


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures = {'cases': make_cases(read_clips())}
    inputs, outputs = OUT/'hast-sec-fixtures.json', OUT/'hast-sec-samples.json'
    inputs.write_text(json.dumps(fixtures, separators=(',', ':')))
    command = ['node', str(ROOT/'scripts/hast_sec_runtime.mjs'), str(inputs), str(outputs)]
    try:
        completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=240)
    except subprocess.CalledProcessError as exc:
        (OUT/'hast-sec-failure.log').write_text((exc.stdout or '')+(exc.stderr or ''))
        raise
    print(completed.stdout, end='')
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end='')
    runtime = json.loads(outputs.read_text())
    write_previews(fixtures, runtime)
    cases = evaluate(fixtures, runtime)
    revision = json.loads((ROOT/'vendor/HAST-SEC/UPSTREAM.json').read_text())['revision']
    report = {'revision': revision, 'fixtureSha256': hashlib.sha256(inputs.read_bytes()).hexdigest(),
              'scope': '原始模块离线对照；无 VRM 视觉验收；未改生产管线和重定向',
              'methodology': runtime['methodology'], 'nativeLibrary': runtime['nativeLibrary'],
              'guards': runtime['guards'], 'timings': runtime['timings'],
              'summary': summarize(cases), 'cases': cases}
    (OUT/'hast-sec-comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    write_report(report)
    (OUT/'hast-sec-comparison.log').write_text(completed.stdout+json.dumps(report['summary'], ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'cases': len(cases), 'summary': report['summary']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()

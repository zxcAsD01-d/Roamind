import * as THREE from 'three';
import type { VRM } from '@pixiv/three-vrm';
import type { Collider } from './house';
import { contactBone, contactRigs } from './contact_rig';
import type { TouchPlan, TouchTarget } from './touch_contact';

const axes = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
export type ContactMeasurement = { error: number; displacement: number; frame: number; samples: number; ms: number;
  contactPhase: boolean; normalGapMeters: number; tangentMeters: number; observedContact: boolean;
  endpoint: number[]; surfacePoint: number[]; sigmaMeters: number };
export const contactWeight = (frame: number, begin: number, end: number, count: number): number => {
  const t = THREE.MathUtils.clamp(frame < begin ? (frame+1)/begin : frame >= end ? (count-frame-1)/(count-end) : 1, 0, 1);
  return t*t*t*(10+t*(-15+6*t));
};

function bounded(before: THREE.Quaternion, after: THREE.Quaternion, limit: number): THREE.Quaternion {
  const angle = before.angleTo(after);
  return before.clone().slerp(after, Math.min(1, limit/Math.max(angle, 1e-8)));
}

/** Position feedback over an already validated clip, with no root translation. */
export class LiveContact {
  private chain: THREE.Object3D[];
  private end: THREE.Object3D;
  private previous: THREE.Quaternion[] | null = null;
  private offset = new THREE.Vector3();
  private latest: TouchTarget;
  private boxes: { id: string; box: THREE.Box3 }[];
  private lastTime = -1;

  constructor(private vrm: VRM, private plan: TouchPlan, private prefix: number, obstacles: Collider[]) {
    const rig = contactRigs[plan.effector.name];
    this.chain = rig.nodes.map(name => contactBone(vrm, name));
    this.end = contactBone(vrm, rig.end); this.latest = plan.touch.target;
    this.boxes = obstacles.map(item => ({ id: item.id,
      box: new THREE.Box3().setFromCenterAndSize(new THREE.Vector3().fromArray(item.center),
        new THREE.Vector3().fromArray(item.size)) }));
  }

  observe(target: TouchTarget): void {
    const initial = this.plan.touch.target;
    if (target.targetId !== initial.targetId || target.personId !== initial.personId || target.frame !== initial.frame) {
      throw new Error('接触目标或坐标系改变');
    }
    if (Date.now()/1000-target.observedAt > 1.5) throw new Error('接触视觉观测过期');
    if (new THREE.Vector3().fromArray(target.position).distanceTo(new THREE.Vector3().fromArray(initial.position)) > .18) {
      throw new Error('接触目标超出局部跟踪范围，需要重新规划');
    }
    if (target.normal.reduce((sum, value, i) => sum+value*initial.normal[i], 0) < .8) throw new Error('目标表面朝向变化过大');
    this.latest = target;
  }

  update(time: number, delta: number): ContactMeasurement | null {
    const frame = time*20-this.prefix;
    if (frame < 0 || frame >= this.plan.frames || time <= this.lastTime) return null;
    const started = performance.now(); this.lastTime = time;
    if (Date.now()/1000-this.latest.observedAt > 1.5) throw new Error('接触视觉观测过期，已停止');
    const baseline = this.chain.map(node => node.quaternion.clone());
    const center = this.end.getWorldPosition(new THREE.Vector3());
    const offset = this.targetOffset(frame);
    this.offset.add(offset.sub(this.offset).clampLength(0, .4*delta));
    const target = center.clone().add(this.offset);
    try {
      this.solve(target, baseline);
      this.chain.forEach((node, index) => {
        if (this.previous) node.quaternion.copy(bounded(this.previous[index], node.quaternion, 3*delta));
        node.updateWorldMatrix(false, true);
      });
      const error = this.end.getWorldPosition(new THREE.Vector3()).distanceTo(target);
      if (!Number.isFinite(error) || error > .06) throw new Error('动态接触求解误差超过 6 cm，需要重新规划');
      this.checkClearance(); this.checkSupport();
      this.previous = this.chain.map(node => node.quaternion.clone());
      return { error, displacement: this.offset.length(), frame, samples: this.latest.trackingSamples ?? 1,
        ms: performance.now()-started, ...this.measureContact(frame) };
    } catch (error) {
      this.chain.forEach((node, index) => node.quaternion.copy(baseline[index]));
      this.chain[0].updateWorldMatrix(false, true);
      throw error;
    }
  }

  private measureContact(frame: number): Omit<ContactMeasurement, 'error' | 'displacement' | 'frame' | 'samples' | 'ms'> {
    const endpoint = this.end.getWorldPosition(new THREE.Vector3());
    const surface = new THREE.Vector3().fromArray(this.latest.position);
    const normal = new THREE.Vector3().fromArray(this.latest.normal).normalize();
    const difference = endpoint.clone().sub(surface), distance = difference.dot(normal);
    const normalGapMeters = distance-this.plan.effector.radius;
    const tangentMeters = difference.addScaledVector(normal, -distance).length();
    const [begin, end] = this.plan.touch.contactFrames;
    const contactPhase = frame >= begin && frame < end;
    return { contactPhase, normalGapMeters, tangentMeters,
      observedContact: contactPhase && Math.abs(normalGapMeters) <= .025 && tangentMeters <= .04,
      endpoint: endpoint.toArray(), surfacePoint: surface.toArray(), sigmaMeters: this.latest.sigmaMeters };
  }

  private targetOffset(frame: number): THREE.Vector3 {
    const initial = this.plan.touch.target, radius = this.plan.effector.radius;
    const offset = new THREE.Vector3().fromArray(this.latest.position).sub(new THREE.Vector3().fromArray(initial.position));
    offset.addScaledVector(new THREE.Vector3().fromArray(this.latest.normal).sub(new THREE.Vector3().fromArray(initial.normal)), radius);
    const [begin, end] = this.plan.touch.contactFrames;
    return offset.multiplyScalar(contactWeight(frame, begin, end, this.plan.frames));
  }

  private solve(target: THREE.Vector3, baseline: THREE.Quaternion[]): void {
    for (let iteration = 0; iteration < 6; iteration++) {
      const current = this.end.getWorldPosition(new THREE.Vector3()), error = target.clone().sub(current);
      if (error.length() < .004) return;
      const columns = this.chain.flatMap(node => {
        const world = node.getWorldQuaternion(new THREE.Quaternion()), position = node.getWorldPosition(new THREE.Vector3());
        return axes.map(axis => axis.clone().applyQuaternion(world).cross(current.clone().sub(position)));
      });
      const normal = new THREE.Matrix3().set(.003, 0, 0, 0, .003, 0, 0, 0, .003);
      for (const c of columns) {
        const outer = [c.x*c.x, c.x*c.y, c.x*c.z, c.y*c.x, c.y*c.y, c.y*c.z, c.z*c.x, c.z*c.y, c.z*c.z];
        normal.elements.forEach((v, i) => { normal.elements[i] = v+outer[i]; });
      }
      const direction = error.applyMatrix3(normal.invert());
      this.chain.forEach((node, index) => {
        const vector = new THREE.Vector3().fromArray(columns.slice(index*3, index*3+3).map(c => c.dot(direction))).clampLength(0, .045);
        const angle = vector.length();
        if (angle > 1e-8) node.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(vector.divideScalar(angle), angle));
        const limit = this.plan.effector.kind === 'arm' ? .35 : .18;
        node.quaternion.copy(bounded(baseline[index], node.quaternion, limit)); node.updateWorldMatrix(false, true);
      });
    }
  }

  private checkClearance(): void {
    const nodes = ['hips', 'spine', 'chest', 'neck', 'head', 'rightUpperArm', 'rightLowerArm', 'rightHand',
      'leftUpperArm', 'leftLowerArm', 'leftHand', 'rightUpperLeg', 'rightLowerLeg', 'rightFoot',
      'leftUpperLeg', 'leftLowerLeg', 'leftFoot'] as const;
    const points = nodes.map(name => contactBone(this.vrm, name));
    for (const node of points) {
      const end = node.getWorldPosition(new THREE.Vector3()), start = node.parent?.getWorldPosition(new THREE.Vector3()) ?? end;
      const steps = Math.max(1, Math.ceil(start.distanceTo(end)/.04));
      for (let step = 0; step <= steps; step++) {
        const sample = start.clone().lerp(end, step/steps);
        const radius = node === this.end ? this.plan.effector.radius : .045;
        const collision = this.boxes.find(item => item.box.distanceToPoint(sample) < radius-.001);
        if (collision) throw new Error(`动态接触骨段碰撞：${collision.id}`);
      }
    }
  }

  private checkSupport(): void {
    if (this.plan.effector.kind !== 'leg') return;
    const side = this.plan.effector.name === 'left_foot' ? 'right' : 'left';
    const foot = contactBone(this.vrm, `${side}Foot`).getWorldPosition(new THREE.Vector3());
    const hips = contactBone(this.vrm, 'hips').getWorldPosition(new THREE.Vector3());
    if (foot.y > .16 || Math.hypot(hips.x-foot.x, hips.z-foot.z) > .10) throw new Error('脚部接触失去支撑姿态');
  }
}

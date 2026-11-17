import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';
import { SmplClip, VrmRetargeter, type SmplMotionJson } from './retarget';
import type { EffectorName } from './contact_rig';

export type TouchTarget = { targetId: string; personId?: string; label?: string; text?: string; requestedPixel?: number[];
  referenceKind?: string; position: number[]; normal: number[];
  tangent: number[]; sigmaMeters: number; observedAt: number; frameId: number; source: string;
  localSurfaceRadius: number; frame: string; velocity?: number[]; trackingSamples?: number;
  filteredSigmaMeters?: number; anatomy?: { estimate: string; part: string; graspVerified: boolean } };
export type TouchPlan = { id: string; duration: number; frames: number; handJoint: number; handTrack: number[][];
  contactTrack: number[][] | null; contactStart: number[] | null;
  handStart: number[]; touch: { target: TouchTarget; clientId: string; contactFrames: number[];
    options: { hand: 'left' | 'right'; effector: EffectorName; operation: string; duration: number; amplitude: number } };
  effector: { name: EffectorName; joint: number; chain: number[]; bone: VRMHumanBoneName; radius: number; kind: string } };
const smooth = (v: number): number => { const t = THREE.MathUtils.clamp(v, 0, 1); return t*t*t*(10+t*(-15+6*t)); };

function writeQuaternion(output: number[], offset: number, value: THREE.Quaternion): void {
  value.normalize(); if (value.w < 0) value.set(-value.x, -value.y, -value.z, -value.w);
  const angle = 2*Math.acos(THREE.MathUtils.clamp(value.w, -1, 1));
  const scale = angle < 1e-7 ? 2 : angle/Math.sin(angle/2);
  output.splice(offset, 3, value.x*scale, value.y*scale, value.z*scale);
}

function quaternion(values: number[], offset: number): THREE.Quaternion {
  const vector = new THREE.Vector3().fromArray(values, offset), angle = vector.length();
  return angle < 1e-8 ? new THREE.Quaternion() : new THREE.Quaternion().setFromAxisAngle(vector.divideScalar(angle), angle);
}

/** Numerical Jacobian uses the unchanged retargeter, so VRM proportions are measured rather than guessed. */
function solveChain(sample: SmplMotionJson, evaluate: () => THREE.Vector3, target: THREE.Vector3, arm: number[]): number {
  const originals = arm.map(joint => quaternion(sample.localRot, joint*3));
  for (let iteration = 0; iteration < 14; iteration++) {
    const current = evaluate(), error = target.clone().sub(current);
    if (error.length() < .018) return error.length();
    const columns = arm.flatMap(joint => [0, 1, 2].map(axis => {
      const offset = joint*3+axis, before = sample.localRot[offset];
      sample.localRot[offset] += .012;
      const derivative = evaluate().sub(current).divideScalar(.012);
      sample.localRot[offset] = before;
      return derivative;
    }));
    const normal = new THREE.Matrix3().set(.004, 0, 0, 0, .004, 0, 0, 0, .004);
    for (const c of columns) {
      const outer = new THREE.Matrix3().set(c.x*c.x, c.x*c.y, c.x*c.z, c.y*c.x, c.y*c.y, c.y*c.z, c.z*c.x, c.z*c.y, c.z*c.z);
      normal.elements.forEach((value, index) => { normal.elements[index] = value+outer.elements[index]; });
    }
    const direction = error.applyMatrix3(normal.invert());
    columns.forEach((c, index) => {
      sample.localRot[arm[Math.floor(index/3)]*3+index%3] += THREE.MathUtils.clamp(c.dot(direction), -.10, .10);
    });
    arm.forEach((joint, index) => {
      const value = quaternion(sample.localRot, joint*3), angle = originals[index].angleTo(value);
      if (angle > .8) value.copy(originals[index]).slerp(quaternion(sample.localRot, joint*3), .8/angle);
      writeQuaternion(sample.localRot, joint*3, value);
    });
  }
  return evaluate().distanceTo(target);
}

function desiredTrack(plan: TouchPlan, actualStart: THREE.Vector3): THREE.Vector3[] {
  const offset = actualStart.clone().sub(new THREE.Vector3().fromArray(plan.contactStart ?? plan.handStart));
  const [begin, end] = plan.touch.contactFrames;
  return (plan.contactTrack ?? plan.handTrack).map((point, frame) => {
    const weight = frame < begin ? 1-smooth((frame+1)/begin) : frame >= end ? smooth((frame-end+1)/(plan.frames-end)) : 0;
    return new THREE.Vector3().fromArray(point).addScaledVector(offset, weight);
  });
}

/** Corrections touch only the selected contact chain; root and original retargeter remain intact. */
export function fitTouchContact(vrm: VRM, root: THREE.Group, motion: SmplMotionJson, plan: TouchPlan, prefix: number): { motion: SmplMotionJson; error: number } {
  const bones = Object.keys(vrm.humanoid.humanBones).map(name => vrm.humanoid.getNormalizedBoneNode(name as VRMHumanBoneName)).filter(node => node !== null);
  const saved = bones.map(bone => bone.quaternion.clone()), savedRoot = root.position.clone();
  const hand = vrm.humanoid.getNormalizedBoneNode(plan.effector.bone);
  if (!hand) throw new Error('VRM 缺少所选接触执行部位骨骼');
  const result = { ...motion, localRot: [...motion.localRot] };
  const sample = { ...motion, frames: 1, duration: .05, localRot: motion.localRot.slice((prefix-1)*81, prefix*81),
    rootDelta: motion.rootDelta.slice((prefix-1)*3, prefix*3), footContacts: undefined };
  const evaluate = (): THREE.Vector3 => {
    const driver = new VrmRetargeter(vrm, SmplClip.from(sample));
    root.position.copy(driver.rootAt(0)); driver.apply(0); root.updateWorldMatrix(true, true);
    return hand.getWorldPosition(new THREE.Vector3());
  };
  try { return fitFrames(result, sample, plan, prefix, evaluate); }
  finally {
    root.position.copy(savedRoot);
    bones.forEach((bone, index) => bone.quaternion.copy(saved[index])); root.updateWorldMatrix(true, true);
  }
}

function fitFrames(result: SmplMotionJson, sample: SmplMotionJson, plan: TouchPlan, prefix: number,
  evaluate: () => THREE.Vector3): { motion: SmplMotionJson; error: number } {
  const targets = desiredTrack(plan, evaluate());
  const arm = plan.effector.chain;
  let largest = 0;
  const previous = arm.map(joint => quaternion(sample.localRot, joint*3));
  for (let frame = 0; frame < plan.frames; frame++) {
    const offset = (prefix+frame)*81;
    sample.localRot = result.localRot.slice(offset, offset+81);
    sample.rootDelta = result.rootDelta.slice((prefix+frame)*3, (prefix+frame+1)*3);
    solveChain(sample, evaluate, targets[frame], arm);
    arm.forEach((joint, index) => {
      const value = quaternion(sample.localRot, joint*3), step = previous[index].angleTo(value);
      if (step > .28) value.copy(previous[index]).slerp(quaternion(sample.localRot, joint*3), .28/step);
      writeQuaternion(sample.localRot, joint*3, value); previous[index].copy(value);
    });
    const error = evaluate().distanceTo(targets[frame]); largest = Math.max(largest, error);
    if (!Number.isFinite(error) || error > .06) throw new Error(`VRM 执行部位轨迹偏差 ${(error*100).toFixed(1)} cm 超过 6 cm，拒绝播放`);
    result.localRot.splice(offset, 81, ...sample.localRot);
  }
  return { motion: result, error: largest };
}

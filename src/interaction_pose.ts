import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';
import { contactBone } from './contact_rig';

export type BonePose = { node: THREE.Object3D; position: THREE.Vector3; rotation: THREE.Quaternion };
export type Pose = BonePose[];
export const smoothStep = (t: number): number => { t = THREE.MathUtils.clamp(t, 0, 1); return t*t*t*(10+t*(-15+6*t)); };
export const point = (vrm: VRM, name: VRMHumanBoneName): THREE.Vector3 => contactBone(vrm, name).getWorldPosition(new THREE.Vector3());
export function capturePose(vrm: VRM): Pose {
  return Object.keys(vrm.humanoid.humanBones).map(name => vrm.humanoid.getNormalizedBoneNode(name as VRMHumanBoneName))
    .filter((node): node is THREE.Object3D => node !== null)
    .map(node => ({ node, position: node.position.clone(), rotation: node.quaternion.clone() }));
}
export function applyPose(pose: Pose): void {
  for (const row of pose) { row.node.position.copy(row.position); row.node.quaternion.copy(row.rotation); }
  pose[0]?.node.updateWorldMatrix(true, true);
}
export function blendPose(from: Pose, to: Pose, t: number): void {
  for (let i = 0; i < to.length; i++) {
    to[i].node.position.lerpVectors(from[i].position, to[i].position, t);
    to[i].node.quaternion.slerpQuaternions(from[i].rotation, to[i].rotation, t);
  }
  to[0]?.node.updateWorldMatrix(true, true);
}
export function setWorldPosition(node: THREE.Object3D, position: THREE.Vector3): void {
  node.position.copy(node.parent ? node.parent.worldToLocal(position.clone()) : position);
  node.updateWorldMatrix(true, true);
}
export function setWorldRotation(node: THREE.Object3D, rotation: THREE.Quaternion): void {
  const parent = node.parent?.getWorldQuaternion(new THREE.Quaternion()) ?? new THREE.Quaternion();
  node.quaternion.copy(parent.invert().multiply(rotation)); node.updateWorldMatrix(true, true);
}
function aim(node: THREE.Object3D, end: THREE.Object3D, target: THREE.Vector3): void {
  const origin = node.getWorldPosition(new THREE.Vector3());
  const from = end.getWorldPosition(new THREE.Vector3()).sub(origin).normalize();
  const to = target.clone().sub(origin).normalize();
  const rotation = new THREE.Quaternion().setFromUnitVectors(from, to).multiply(node.getWorldQuaternion(new THREE.Quaternion()));
  setWorldRotation(node, rotation);
}
/** Analytic two-bone IK. Lengths are measured on the current avatar, never fixed SMPL dimensions. */
export function solveLimb(vrm: VRM, names: [VRMHumanBoneName, VRMHumanBoneName, VRMHumanBoneName],
  target: THREE.Vector3, pole: THREE.Vector3): number {
  const [base, middle, end] = names.map(name => contactBone(vrm, name));
  const a = base.getWorldPosition(new THREE.Vector3()), b = middle.getWorldPosition(new THREE.Vector3());
  const c = end.getWorldPosition(new THREE.Vector3()), l1 = a.distanceTo(b), l2 = b.distanceTo(c);
  const delta = target.clone().sub(a), requested = delta.length();
  const distance = THREE.MathUtils.clamp(requested, Math.abs(l1-l2)+.0001, l1+l2-.0001);
  const direction = delta.normalize();
  const bend = pole.clone().addScaledVector(direction, -pole.dot(direction)).normalize();
  const along = (l1*l1-l2*l2+distance*distance)/(2*distance);
  const knee = a.clone().addScaledVector(direction, along).addScaledVector(bend, Math.sqrt(Math.max(0, l1*l1-along*along)));
  aim(base, middle, knee); aim(middle, end, target);
  return end.getWorldPosition(new THREE.Vector3()).distanceTo(target);
}
export function plantFeet(vrm: VRM, feet: THREE.Vector3[], forward: THREE.Vector3, rotations?: THREE.Quaternion[]): number {
  return Math.max(...(['left', 'right'] as const).map((side, i) => {
    const error = solveLimb(vrm, [`${side}UpperLeg`, `${side}LowerLeg`, `${side}Foot`], feet[i], forward);
    if (rotations) setWorldRotation(contactBone(vrm, `${side}Foot`), rotations[i]);
    return error;
  }));
}

/** The wrist joint is not a grasp point: locate the palm from the avatar's knuckles. */
export function palmPosition(vrm: VRM, side: 'left' | 'right' = 'right'): THREE.Vector3 {
  const wrist = point(vrm, `${side}Hand`);
  const knuckles = ([`${side}IndexProximal`, `${side}MiddleProximal`, `${side}LittleProximal`] as const)
    .map(name => vrm.humanoid.getNormalizedBoneNode(name)).filter((node): node is THREE.Object3D => node !== null);
  if (!knuckles.length) throw new Error('拿取需要可标定掌心的手指骨骼');
  const center = knuckles.reduce((sum, node) => sum.add(node.getWorldPosition(new THREE.Vector3())), new THREE.Vector3()).divideScalar(knuckles.length);
  return wrist.lerp(center, .65);
}

export function reachPalm(vrm: VRM, target: THREE.Vector3, pole: THREE.Vector3, side: 'left' | 'right' = 'right', rotation?: THREE.Quaternion): number {
  for (let iteration = 0; iteration < 8; iteration++) {
    const offset = palmPosition(vrm, side).sub(point(vrm, `${side}Hand`));
    solveLimb(vrm, [`${side}UpperArm`, `${side}LowerArm`, `${side}Hand`], target.clone().sub(offset), pole);
    if (rotation) setWorldRotation(contactBone(vrm, `${side}Hand`), rotation);
    if (palmPosition(vrm, side).distanceTo(target) < .003) break;
  }
  return palmPosition(vrm, side).distanceTo(target);
}

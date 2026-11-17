import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';

export type EffectorName = 'right_hand' | 'left_hand' | 'right_foot' | 'left_foot' | 'head';
const collisionProfiles = new WeakMap<VRM, { radius: number; height: number }>();
type Rig = { nodes: VRMHumanBoneName[]; end: VRMHumanBoneName; kind: 'arm' | 'leg' | 'axial' };
export const contactRigs: Record<EffectorName, Rig> = {
  right_hand: { nodes: ['rightUpperArm', 'rightLowerArm'], end: 'rightHand', kind: 'arm' },
  left_hand: { nodes: ['leftUpperArm', 'leftLowerArm'], end: 'leftHand', kind: 'arm' },
  right_foot: { nodes: ['rightUpperLeg', 'rightLowerLeg'], end: 'rightFoot', kind: 'leg' },
  left_foot: { nodes: ['leftUpperLeg', 'leftLowerLeg'], end: 'leftFoot', kind: 'leg' },
  head: { nodes: ['chest', 'neck'], end: 'head', kind: 'axial' },
};

export function contactBone(vrm: VRM, name: VRMHumanBoneName): THREE.Object3D {
  const bone = vrm.humanoid.getNormalizedBoneNode(name);
  if (!bone) throw new Error(`自身接触执行链缺少骨骼 ${name}`);
  return bone;
}

export function selfEffector(vrm: VRM, name: EffectorName): { base: number[]; middle: number[]; end: number[] } {
  const rig = contactRigs[name];
  if (!rig) throw new Error('不支持的接触执行部位');
  const position = (bone: VRMHumanBoneName): number[] => contactBone(vrm, bone).getWorldPosition(new THREE.Vector3()).toArray();
  return { base: position(rig.nodes[0]), middle: position(rig.nodes.at(-1)!), end: position(rig.end) };
}

/** Proprioception only: no object positions or scene map enter the planning profile. */
export function bodyReachProfile(vrm: VRM, root: THREE.Object3D): {
  support: { legReach: number; footHeight: number; footOffset: number };
  stature: number; effectors: Record<string, { reach: number; height: number; lateral: number }>;
} {
  const origin = root.getWorldPosition(new THREE.Vector3());
  const effectors: Record<string, { reach: number; height: number; lateral: number }> = {};
  for (const name of ['left_hand', 'right_hand'] as const) {
    const own = selfEffector(vrm, name);
    const base = new THREE.Vector3().fromArray(own.base), middle = new THREE.Vector3().fromArray(own.middle);
    effectors[name] = { reach: base.distanceTo(middle)+middle.distanceTo(new THREE.Vector3().fromArray(own.end)),
      height: base.y-origin.y, lateral: Math.hypot(base.x-origin.x, base.z-origin.z) };
  }
  const feet = (['left_foot', 'right_foot'] as const).map(name => selfEffector(vrm, name));
  const support = { legReach: Math.min(...feet.map(foot => new THREE.Vector3().fromArray(foot.base).distanceTo(new THREE.Vector3().fromArray(foot.middle))
      +new THREE.Vector3().fromArray(foot.middle).distanceTo(new THREE.Vector3().fromArray(foot.end)))),
    footHeight: Math.min(...feet.map(foot => foot.end[1]-origin.y)),
    footOffset: Math.max(...feet.map(foot => Math.hypot(foot.end[0]-origin.x, foot.end[2]-origin.z))) };
  return { effectors, support, stature: bodyCollisionProfile(vrm).height };
}

export function bodyCollisionProfile(vrm: VRM): { radius: number; height: number } {
  const calibrated = collisionProfiles.get(vrm);
  if (calibrated) return { ...calibrated };
  const position = (name: VRMHumanBoneName): THREE.Vector3 => contactBone(vrm, name).getWorldPosition(new THREE.Vector3());
  const shoulder = position('leftUpperArm').distanceTo(position('rightUpperArm'));
  const hips = position('leftUpperLeg').distanceTo(position('rightUpperLeg'));
  const feet = Math.min(position('leftFoot').y, position('rightFoot').y);
  const profile = { radius: Math.max(shoulder/2+.045, hips/2+.065), height: position('head').y-feet+.105 };
  collisionProfiles.set(vrm, profile);
  return { ...profile };
}

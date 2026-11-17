import * as THREE from 'three';
import type { VRM } from '@pixiv/three-vrm';
import { contactBone } from './contact_rig';
import { measureVrmRest } from './retarget';
import { capturePose, palmPosition, point, reachPalm, setWorldRotation, type Pose } from './interaction_pose';
import type { HeldProp } from './interaction_props';

export type ConsumptionState = { commandId: string; objectId: string; operation: 'eat' | 'drink';
  completed: boolean; returned: boolean; mouthError: number; graspError: number; remaining: number; variant: number };
export const CONSUMPTION_STYLES = [
  { name: '从容', lift: 1.35, contact: .85, pause: .8, tilt: .48 },
  { name: '小口', lift: 1.15, contact: .7, pause: .65, tilt: .38 },
  { name: '仔细', lift: 1.5, contact: 1., pause: 1., tilt: .55 },
] as const;

function bodyOrientation(vrm: VRM, bone: 'head' | 'chest'): THREE.Quaternion {
  return contactBone(vrm, bone).getWorldQuaternion(new THREE.Quaternion())
    .multiply(measureVrmRest(vrm).get(bone)!.clone().invert());
}

/** Scale the mouth estimate by the avatar's neck/head geometry rather than scene coordinates. */
export function mouthPosition(vrm: VRM): THREE.Vector3 {
  const left = vrm.humanoid.getNormalizedBoneNode('leftEye'), right = vrm.humanoid.getNormalizedBoneNode('rightEye');
  if (left && right) {
    const eyes = left.getWorldPosition(new THREE.Vector3()).add(right.getWorldPosition(new THREE.Vector3())).multiplyScalar(.5);
    const orientation = bodyOrientation(vrm, 'head'), up = new THREE.Vector3(0, 1, 0).applyQuaternion(orientation);
    const height = eyes.clone().sub(point(vrm, 'head')).dot(up);
    if (height > .03 && height < .25) return eyes.addScaledVector(up, -height*.45)
      .add(new THREE.Vector3(0, 0, .012).applyQuaternion(orientation));
  }
  const length = Math.max(.08, point(vrm, 'head').distanceTo(point(vrm, 'neck')));
  return new THREE.Vector3(0, length*.35, length*.72).applyQuaternion(bodyOrientation(vrm, 'head')).add(point(vrm, 'head'));
}

export function itemContact(item: HeldProp, operation: 'eat' | 'drink'): THREE.Vector3 {
  return operation === 'drink' ? new THREE.Vector3(0, item.definition.size[1]/2, -item.definition.size[0]*.35)
    : new THREE.Vector3(0, 0, -item.definition.size[2]*.43);
}

export function consumptionPose(vrm: VRM, item: HeldProp, operation: 'eat' | 'drink', tilt: number): { pose: Pose; error: number } {
  const orientation = bodyOrientation(vrm, 'chest');
  if (operation === 'drink') orientation.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -tilt));
  const handRotation = orientation.multiply(item.mesh.quaternion.clone().invert());
  setWorldRotation(contactBone(vrm, 'rightHand'), handRotation);
  item.mesh.updateWorldMatrix(true, true);
  const contact = item.mesh.localToWorld(itemContact(item, operation));
  const target = mouthPosition(vrm).sub(contact.sub(palmPosition(vrm)));
  const pole = new THREE.Vector3(-.8, -.6, -.15).applyQuaternion(bodyOrientation(vrm, 'chest'));
  reachPalm(vrm, target, pole, 'right', handRotation);
  item.mesh.updateWorldMatrix(true, true);
  const error = item.mesh.localToWorld(itemContact(item, operation)).distanceTo(mouthPosition(vrm));
  if (!Number.isFinite(error) || error > .035) throw new Error('口部目标超出当前手臂范围，拒绝虚假吃喝');
  return { pose: capturePose(vrm), error };
}

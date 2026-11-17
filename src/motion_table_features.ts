import { Quaternion, Vector3 } from 'three';
import type { SmplClip } from './retarget';

export const TABLE_JOINTS = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26];
export const TABLE_ANGULAR_OFFSET = 5+TABLE_JOINTS.length*4;
export const TABLE_FEATURE_SIZE = TABLE_ANGULAR_OFFSET+TABLE_JOINTS.length*3;
export const tableYaw = (q: Quaternion): number => {
  const v = new Vector3(0, 0, 1).applyQuaternion(q);
  return Math.atan2(v.x, v.z);
};
export const wrapYaw = (v: number): number => Math.atan2(Math.sin(v), Math.cos(v));
export const yawRotation = (v: number): Quaternion => new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), v);
export function rotationVector(q: Quaternion): Vector3 {
  const sign = q.w < 0 ? -1 : 1, w = Math.min(1, Math.max(-1, q.w*sign));
  const angle = 2*Math.acos(w), s = Math.sqrt(Math.max(0, 1-w*w));
  return new Vector3(q.x, q.y, q.z).multiplyScalar(sign*(s < 1e-7 ? 2 : angle/s));
}
export function vectorRotation(v: Vector3): Quaternion {
  const angle = v.length();
  return angle < 1e-8 ? new Quaternion() : new Quaternion().setFromAxisAngle(v.clone().divideScalar(angle), angle);
}
export function tableRotation(clip: SmplClip, frame: number, joint: number): Quaternion {
  const a = Math.max(0, Math.min(clip.frames-1, Math.floor(frame))), b = Math.min(clip.frames-1, a+1);
  return new Quaternion().fromArray(clip.local, (a*27+joint)*4)
    .slerp(new Quaternion().fromArray(clip.local, (b*27+joint)*4), frame-a);
}

export function tableBoundary(clip: SmplClip, frame = clip.frames-1): number[] {
  const before = Math.max(0, frame-1), after = Math.min(clip.frames-1, frame+1), dt = Math.max(1, after-before)/20;
  const q = tableRotation(clip, frame, 0), inverseYaw = yawRotation(-tableYaw(q));
  const velocity = clip.rootAt(after, new Vector3()).sub(clip.rootAt(before, new Vector3())).divideScalar(dt).applyQuaternion(inverseYaw);
  const contact = clip.motion.footContacts?.slice(frame*4, frame*4+4) ?? [0, 0, 0, 0];
  const poses = TABLE_JOINTS.flatMap(joint => {
    const pose = tableRotation(clip, frame, joint);
    return (joint === 0 ? pose.premultiply(inverseYaw) : pose).toArray();
  });
  const angular = TABLE_JOINTS.flatMap(joint => rotationVector(tableRotation(clip, before, joint).invert()
    .multiply(tableRotation(clip, after, joint))).divideScalar(dt).toArray());
  const yawRate = wrapYaw(tableYaw(tableRotation(clip, after, 0))-tableYaw(tableRotation(clip, before, 0)))/dt;
  return [velocity.x, velocity.z, yawRate, Math.max(...contact.slice(0, 2)), Math.max(...contact.slice(2)), ...poses, ...angular];
}

export function scaledBoundary(feature: number[], scale: number, rate: number): number[] {
  return feature.map((v, index) => index < 2 ? v*scale*rate : index === 2 || index >= TABLE_ANGULAR_OFFSET ? v*rate : v);
}

export function tableCost(a: number[], b: number[]): { cost: number; rms: number; maximum: number; velocity: number } {
  let pose = 0, maximum = 0, angular = 0;
  for (let joint = 0; joint < TABLE_JOINTS.length; joint++) {
    const k = 5+joint*4;
    const dot = Math.abs(a[k]*b[k]+a[k+1]*b[k+1]+a[k+2]*b[k+2]+a[k+3]*b[k+3]);
    const angle = 2*Math.acos(Math.min(1, dot));
    pose += angle*angle/TABLE_JOINTS.length; maximum = Math.max(maximum, angle);
  }
  for (let k = TABLE_ANGULAR_OFFSET; k < a.length; k++) angular += (a[k]-b[k])**2/(a.length-TABLE_ANGULAR_OFFSET);
  const velocity = Math.hypot(a[0]-b[0], a[1]-b[1]);
  const contact = (Math.abs(a[3]-b[3])+Math.abs(a[4]-b[4]))/2;
  return { cost: pose+.4*velocity**2+.02*angular+.2*contact+.04*(a[2]-b[2])**2,
    rms: Math.sqrt(pose), maximum, velocity };
}

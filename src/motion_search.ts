import * as THREE from 'three';
import type { SmplMotionJson } from './retarget';

export type FrameIndex = { version: number; entries: number[][] };
export type SearchMotion = SmplMotionJson & { search?: FrameIndex };
const JOINTS = [19, 20, 21, 23, 24, 25];
const UP = new THREE.Vector3(0, 1, 0);

function rotation(motion: SmplMotionJson, frame: number, joint: number): THREE.Quaternion {
  const offset = (frame*27+joint)*3;
  const axis = new THREE.Vector3().fromArray(motion.localRot, offset), angle = axis.length();
  return angle < 1e-8 ? new THREE.Quaternion()
    : new THREE.Quaternion().setFromAxisAngle(axis.divideScalar(angle), angle);
}

function angularVelocity(before: THREE.Quaternion, after: THREE.Quaternion, dt: number): number[] {
  const delta = before.clone().invert().multiply(after).normalize();
  if (delta.w < 0) delta.set(-delta.x, -delta.y, -delta.z, -delta.w);
  const angle = 2*Math.acos(THREE.MathUtils.clamp(delta.w, -1, 1));
  const factor = angle < 1e-6 ? 2/dt : angle/(Math.sin(angle/2)*dt);
  return [delta.x*factor, delta.y*factor, delta.z*factor];
}

/** Same 46 dimensions as the offline AMASS frame index. No scene truth. */
export function boundaryFeatures(motion: SmplMotionJson, frame = motion.frames-1): number[] {
  const before = Math.max(0, frame-1), dt = Math.max(1, frame-before)/motion.fps;
  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(rotation(motion, frame, 0));
  const velocity = new THREE.Vector3().fromArray(motion.rootDelta, frame*3)
    .sub(new THREE.Vector3().fromArray(motion.rootDelta, before*3)).divideScalar(dt)
    .applyAxisAngle(UP, -Math.atan2(forward.x, forward.z));
  const contact = motion.footContacts?.slice(frame*4, frame*4+4) ?? [0, 0, 0, 0];
  const poses = JOINTS.map(joint => rotation(motion, frame, joint));
  return [velocity.x, velocity.z, Math.max(...contact.slice(0, 2)), Math.max(...contact.slice(2)),
    ...poses.flatMap(q => q.toArray()),
    ...poses.flatMap((q, index) => angularVelocity(rotation(motion, before, JOINTS[index]), q, dt))];
}

export function featureCost(query: number[], target: number[], rate: number, scale: number): number {
  let pose = 0, angular = 0;
  for (let joint = 0; joint < 6; joint++) {
    const offset = 4+joint*4;
    let dot = 0;
    for (let axis = 0; axis < 4; axis++) dot += query[offset+axis]*target[offset+axis];
    pose += (2*Math.acos(Math.min(1, Math.abs(dot))))**2/6;
  }
  for (let axis = 28; axis < 46; axis++) angular += (query[axis]-target[axis]*rate)**2/18;
  const speed = (query[0]-target[0]*rate*scale)**2+(query[1]-target[1]*rate*scale)**2;
  const contacts = (Math.abs(query[2]-target[2])+Math.abs(query[3]-target[3]))/2;
  return pose+.3*speed+.15*contacts+.015*angular;
}

/** Offline features avoid rebuilding every candidate skeleton on each query. */
export class MotionSearch {
  private entries: { frame: number; features: number[] }[];
  queries = 0;

  constructor(private motion: SearchMotion) {
    const saved = motion.search;
    if (saved && (saved.version !== 1 || !Array.isArray(saved.entries) || !saved.entries.length
      || saved.entries.some(row => row.length !== 47 || !row.every(Number.isFinite)
        || !Number.isInteger(row[0]) || row[0] < 0 || row[0] >= motion.frames))) {
      throw new Error('AMASS 检索索引无效');
    }
    this.entries = saved ? saved.entries.map(row => ({ frame: row[0], features: row.slice(1) }))
      : Array.from({ length: motion.frames }, (_, frame) => ({ frame, features: boundaryFeatures(motion, frame) }));
  }

  shortlist(query: number[], begin: number, end: number, rate: number, scale: number): { frame: number; cost: number }[] {
    this.queries++;
    const best: { frame: number; cost: number }[] = [];
    for (const entry of this.entries) {
      if (entry.frame < begin || entry.frame > end) continue;
      const cost = featureCost(query, entry.features, rate, scale);
      if (best.length === 6 && cost >= best[5].cost) continue;
      best.push({ frame: entry.frame, cost });
      best.sort((a, b) => a.cost-b.cost);
      if (best.length > 6) best.pop();
    }
    // Narrow entry windows can fall between the offline two-frame samples.
    if (!best.length && begin >= 0 && begin <= end && begin < this.motion.frames) {
      best.push({ frame: begin, cost: featureCost(query, boundaryFeatures(this.motion, begin), rate, scale) });
    }
    return best;
  }
}

import * as THREE from 'three';
import { SmplClip, type SmplMotionJson } from './retarget';

export type PlaybackHistory = { motion: SmplMotionJson; time: number };

/** Keep the played poses plus a short lead; discard the superseded future. */
export function capturePlaybackHistory(
  clip: SmplClip, time: number, playing: boolean, speed: number,
): PlaybackHistory {
  if (clip.zUp || clip.joints !== 27 || clip.fps !== 20) {
    throw new Error('连续生成需要先选择 ARDY 27 骨骼、20 fps 动作');
  }
  const current = Math.min(clip.frames - 1, Math.floor(time * clip.fps));
  const lead = playing ? Math.ceil(1.2 * speed * clip.fps) : 0;
  const end = Math.min(clip.frames, current + lead + 1);
  const start = Math.max(0, end - 160);
  return {
    time: Math.max(0, time - start / clip.fps),
    motion: {
      ...clip.motion, frames: end - start, duration: (end - start) / clip.fps,
      localRot: clip.motion.localRot.slice(start * clip.joints * 3, end * clip.joints * 3),
      rootDelta: clip.motion.rootDelta.slice(start * 3, end * 3),
      rootOrigin: clip.motion.rootOrigin ?? [0, 0.92, 0],
      footContacts: clip.motion.footContacts?.slice(start*4, end*4),
    },
  };
}

function quaternionAt(clip: SmplClip, frame: number, joint: number): THREE.Quaternion {
  return new THREE.Quaternion().fromArray(clip.local, (frame * clip.joints + joint) * 4);
}

function writeRotation(target: number[], rotation: THREE.Quaternion): void {
  rotation.normalize();
  if (rotation.w < 0) rotation.set(-rotation.x, -rotation.y, -rotation.z, -rotation.w);
  const angle = 2 * Math.acos(THREE.MathUtils.clamp(rotation.w, -1, 1));
  const scale = angle < 1e-8 ? 2 : angle / Math.sin(angle / 2);
  target.push(rotation.x * scale, rotation.y * scale, rotation.z * scale);
}

function heading(rotation: THREE.Quaternion): number {
  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(rotation);
  return Math.atan2(forward.x, forward.z);
}

/** Align a saved clip in the same world, then blend from the running pose. */
export function joinSavedMotion(prefix: SmplMotionJson, next: SmplClip): SmplMotionJson {
  const old = SmplClip.from(prefix);
  if (next.fps !== old.fps || next.joints !== old.joints || next.zUp !== old.zUp
      || next.motion.bones.some((bone, i) => bone !== prefix.bones[i])
      || next.parents.some((parent, i) => parent !== prefix.parents[i])) {
    throw new Error('连贯切换需要相同帧率和骨骼结构的动作');
  }
  const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0),
    heading(quaternionAt(old, old.frames - 1, 0)) - heading(quaternionAt(next, 0, 0)));
  const origin = old.rootAt(old.frames - 1, new THREE.Vector3());
  const previousOrigin = old.rootAt(Math.max(0, old.frames - 2), new THREE.Vector3());
  const rootVelocity = origin.clone().sub(previousOrigin);
  const start = next.rootAt(0, new THREE.Vector3());
  const blendFrames = Math.min(12, next.frames);
  const localRot = [...prefix.localRot];
  const rootDelta = [...prefix.rootDelta];
  for (let frame = 0; frame < next.frames; frame++) {
    const t = blendFrames <= 1 ? 1 : Math.min(1, frame / (blendFrames - 1));
    const weight = t * t * (3 - 2 * t);
    for (let joint = 0; joint < next.joints; joint++) {
      const rotation = quaternionAt(next, frame, joint);
      if (joint === 0) rotation.premultiply(yaw);
      const last = quaternionAt(old, old.frames - 1, joint);
      const before = quaternionAt(old, Math.max(0, old.frames - 2), joint);
      const step = before.invert().multiply(last);
      const predicted = last.clone().multiply(step);
      writeRotation(localRot, predicted.slerp(rotation, weight));
    }
    const target = next.rootAt(frame, new THREE.Vector3()).sub(start).applyQuaternion(yaw).add(origin);
    const predicted = origin.clone().addScaledVector(rootVelocity, frame + 1);
    const root = predicted.lerp(target, weight);
    rootDelta.push(root.x, root.y, root.z);
  }
  const frames = old.frames + next.frames;
  const footContacts = prefix.footContacts || next.motion.footContacts
    ? [...(prefix.footContacts ?? new Array<number>(old.frames*4).fill(0)),
      ...(next.motion.footContacts ?? new Array<number>(next.frames*4).fill(0))] : undefined;
  return { ...next.motion, frames, duration: frames / next.fps, localRot, rootDelta, rootOrigin: prefix.rootOrigin, footContacts };
}

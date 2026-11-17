import * as THREE from 'three';
import { presetSeed, type MotionPhase, type MotionSeed } from './locomotion';
import type { SmplClip } from './retarget';

/** One immutable boundary supplies both the planner and the installed clip. */
export type PresetBoundary = { clip: SmplClip; capturedTime: number; seed: MotionSeed;
  start: number[]; heading: number };
export class PresetBoundaryExpiredError extends Error {}

function continuationLead(time: number, phases: MotionPhase[]): { minimum: number; maximum: number } | undefined {
  const phase = phases.find(item => item.start <= time && item.end > time);
  if (!phase || !['start', 'walk'].includes(phase.kind)) return undefined;
  const brake = phases.find(item => item.kind === 'stop' && item.start > time);
  const available = brake ? Math.max(0, brake.start-time-.05) : .85;
  return { minimum: Math.min(.65, available), maximum: Math.min(.85, available) };
}

export function capturePresetBoundary(clip: SmplClip, time: number, playing: boolean,
  offset: THREE.Vector3, phases: MotionPhase[], retainTail = false, continuation = false): PresetBoundary {
  const seed = presetSeed(clip, time, playing, offset, retainTail,
    continuation ? continuationLead(time, phases) : undefined);
  const current = Math.max(0, Math.min(clip.frames-1, Math.floor(time*clip.fps+1e-6)));
  const frame = Math.max(0, current-40)+seed.motion.frames-1;
  const boundaryTime = frame/clip.fps;
  seed.phase = phases.find(phase => phase.start <= boundaryTime && phase.end > boundaryTime)?.kind
    ?? [...phases].reverse().find(phase => phase.start <= boundaryTime && phase.end >= boundaryTime-.001)?.kind;
  const last = seed.motion.rootDelta.slice(-3);
  const q = new THREE.Quaternion().fromArray(clip.local, frame*clip.joints*4);
  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(q.normalize());
  return { clip, capturedTime: time, seed, start: [last[0], last[1]+.92, last[2]],
    heading: Math.atan2(forward.x, forward.z) };
}

export function advancePresetBoundary(boundary: PresetBoundary, clip: SmplClip | null, time: number): MotionSeed {
  if (clip !== boundary.clip) throw new Error('规划期间动作源已改变，请重新规划');
  const seed = { ...boundary.seed, time: boundary.seed.time+Math.max(0, time-boundary.capturedTime) };
  const end = (seed.motion.frames-1)/seed.motion.fps;
  // The source can finish while the request is in flight. Its retained final
  // pose remains valid; only overrunning an intermediate cut is an expiry.
  const sourceEnd = (boundary.clip.frames-1)/boundary.clip.fps;
  const retainedEnd = boundary.capturedTime+end-boundary.seed.time;
  if (time > sourceEnd+.051) throw new PresetBoundaryExpiredError('动作时间超出原片段尾帧');
  if (seed.time > end+.001 && retainedEnd < sourceEnd-.001)
    throw new PresetBoundaryExpiredError('规划准备超过保留的动作前缀，当前动作继续完成后重新规划');
  seed.time = Math.min(seed.time, end);
  return seed;
}

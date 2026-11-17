import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';
import { SmplClip, VrmRetargeter, type SmplMotionJson } from './retarget';
import type { MotionPhase } from './locomotion';

export type SupportQuality = { samples: number; mean: number; p95: number; measuredFrames: number; floor: number };
const FEET = ['leftFoot', 'leftToes', 'rightFoot', 'rightToes'] as const;
const quantile = (values: number[], p: number): number => [...values].sort((a, b) => a-b)[Math.floor((values.length-1)*p)];

function sampleFrames(clip: SmplClip, phases: MotionPhase[], begin: number, dense: boolean): number[] {
  if (dense || clip.frames-begin <= 180) return Array.from({ length: clip.frames-Math.max(0, begin-1) }, (_, i) => Math.max(0, begin-1)+i);
  const frames = new Set<number>();
  for (let frame = begin; frame < clip.frames; frame += 4) frames.add(frame);
  for (const phase of phases) {
    const start = Math.max(begin, Math.round(phase.start*20)), end = Math.min(clip.frames, Math.round(phase.end*20));
    for (let frame = start; frame < Math.min(end, start+17); frame++) frames.add(frame);
    for (let frame = Math.max(start, end-8); frame < end; frame++) frames.add(frame);
  }
  frames.add(Math.max(0, begin-1));
  return [...frames].sort((a, b) => a-b);
}

function captureFeet(vrm: VRM, clip: SmplClip, frames: number[]): THREE.Vector3[][] {
  const bones = Object.keys(vrm.humanoid.humanBones).map(name =>
    vrm.humanoid.getNormalizedBoneNode(name as VRMHumanBoneName)).filter(node => node !== null);
  const saved = bones.map(bone => bone.quaternion.clone()), driver = new VrmRetargeter(vrm, clip);
  const feet = FEET.map(name => vrm.humanoid.getNormalizedBoneNode(name));
  if (feet.some(node => !node)) throw new Error('动作校验缺少脚部骨骼');
  try {
    return frames.map(frame => {
      driver.apply(frame);
      const root = driver.rootAt(frame);
      return feet.map(node => node!.getWorldPosition(new THREE.Vector3()).add(root));
    });
  } finally {
    bones.forEach((bone, index) => { bone.quaternion.copy(saved[index]); });
    vrm.scene.updateWorldMatrix(true, true);
  }
}

/** Measure geometry on the loaded avatar; contact labels cannot lower the score. */
export function measureSupport(vrm: VRM, motion: SmplMotionJson, phases: MotionPhase[], begin: number, dense = false): SupportQuality {
  const clip = SmplClip.from(motion), frames = sampleFrames(clip, phases, begin, dense), samples = captureFeet(vrm, clip, frames);
  const floors = FEET.map((_, foot) => quantile(samples.map(frame => frame[foot].y), .2)), speed: number[] = [];
  for (let index = 1; index < samples.length; index++) {
    const dt = (frames[index]-frames[index-1])/20;
    for (let foot = 0; foot < 4; foot++) {
      const a = samples[index-1][foot], b = samples[index][foot];
      if (Math.max(a.y, b.y) < floors[foot]+.012 && Math.abs(b.y-a.y)/dt < .08)
        speed.push(Math.hypot(b.x-a.x, b.z-a.z)/dt);
    }
  }
  return { samples: speed.length, mean: speed.length ? speed.reduce((a, b) => a+b, 0)/speed.length : Infinity,
    p95: speed.length ? quantile(speed, .95) : Infinity, measuredFrames: frames.length, floor: Math.min(...floors) };
}

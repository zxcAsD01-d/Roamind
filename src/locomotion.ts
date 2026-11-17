import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';
import { SmplClip, VrmRetargeter, type SmplMotionJson } from './retarget';
import { VariantMemory } from './motion_selection';
import { measureSupport, type SupportQuality } from './motion_quality';
import { MotionSearch, boundaryFeatures, type FrameIndex } from './motion_search';
import { MotionTableData } from './motion_table_data';
import { TableCoverageError } from './motion_table_search';
import { TableCollisionError } from './motion_table_playback';
import { compileTableAction } from './motion_table_action';
import { compileMotionTable } from './motion_table_runtime';
import type { TableDiagnostics } from './motion_table_types';

type Kind = 'idle' | 'start' | 'walk' | 'stop' | 'turn';
type Preset = SmplMotionJson & { kind: Kind; angle: number; distance: number; sourceHeight?: number; heightOffset?: number; family?: string; search?: FrameIndex };
type Asset = { motion: Preset; clip: SmplClip; scale: number; search: MotionSearch };
export type LocomotionStep = { action: string; start: number; end: number; distance: number;
  path?: number[][]; heading?: number; fixedHeading?: boolean; continuous?: boolean };
export type LocomotionPlan = { id: string; duration: number; path: number[][]; timeline: LocomotionStep[]; timing?: 'strict' | 'flexible' };
export type MotionPhase = { kind: Kind | 'action'; start: number; end: number; source: string; variant?: string };
export type PresetResult = { motion: SmplMotionJson; phases: MotionPhase[]; seconds: number; table?: TableDiagnostics };
export type MotionSeed = { motion: SmplMotionJson; time: number; phase?: MotionPhase['kind'] };
type SeedLead = { minimum: number; maximum: number };
type SafeRoot = (previous: THREE.Vector3, next: THREE.Vector3) => string | null;
type Part = { asset: Asset; begin: number; end: number; scale: number; steer?: boolean };
const UP = new THREE.Vector3(0, 1, 0);
const FOOT_NAMES = ['leftFoot', 'leftToes', 'rightFoot', 'rightToes'] as const;
const LEG_JOINTS = [19, 20, 21, 22, 23, 24, 25, 26];
export const PRESET_ACTIONS = new Set(['walk', 'forward', 'turn_left', 'turn_right', 'observe']);
export class PresetUnavailableError extends Error {}
export class PresetCollisionError extends Error {
  constructor(message: string, readonly assetName: string) { super(message); }
}

const wrapped = (angle: number): number => Math.atan2(Math.sin(angle), Math.cos(angle));
const smooth = (t: number): number => t*t*t*(10+t*(-15+6*t));
const yawOf = (q: THREE.Quaternion): number => {
  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
  return Math.atan2(forward.x, forward.z);
};
const rotationAt = (clip: SmplClip, frame: number, joint: number): THREE.Quaternion => {
  const first = Math.floor(frame), next = Math.min(clip.frames-1, first+1);
  return new THREE.Quaternion().fromArray(clip.local, (first*clip.joints+joint)*4)
    .slerp(new THREE.Quaternion().fromArray(clip.local, (next*clip.joints+joint)*4), frame-first);
};

function writeRotation(output: number[], q: THREE.Quaternion): void {
  q.normalize();
  if (q.w < 0) q.set(-q.x, -q.y, -q.z, -q.w);
  const angle = 2*Math.acos(THREE.MathUtils.clamp(q.w, -1, 1));
  const factor = angle < 1e-7 ? 2 : angle/Math.sin(angle/2);
  output.push(q.x*factor, q.y*factor, q.z*factor);
}

function validStop(asset: Asset): boolean {
    if (asset.motion.kind !== 'stop') return true;
    const begin = Math.min(18, asset.clip.frames-10);
    const delta = asset.clip.rootAt(asset.clip.frames-1, new THREE.Vector3())
      .sub(asset.clip.rootAt(begin, new THREE.Vector3())).multiplyScalar(asset.scale);
    return Math.abs(delta.x) <= .08;
}

/** A rigid root-motion scale measured from actual VRM stance motion, never IK. */
function calibrateStride(vrm: VRM, clip: SmplClip): number | null {
  const nodes = FOOT_NAMES.map(name => vrm.humanoid.getNormalizedBoneNode(name));
  if (nodes.some(node => !node)) throw new Error('预设步态需要完整的脚部骨骼');
  const bones = Object.keys(vrm.humanoid.humanBones).map(name =>
    vrm.humanoid.getNormalizedBoneNode(name as VRMHumanBoneName)).filter(node => node !== null);
  const saved = bones.map(bone => bone.quaternion.clone());
  const driver = new VrmRetargeter(vrm, clip), positions: THREE.Vector3[][] = [];
  try {
    for (let frame = 0; frame < clip.frames; frame++) {
      driver.apply(frame);
      positions.push(nodes.map(node => node!.getWorldPosition(new THREE.Vector3())));
    }
  } finally {
    bones.forEach((bone, index) => { bone.quaternion.copy(saved[index]); });
    vrm.scene.updateWorldMatrix(true, true);
  }
  const floors = nodes.map((_, foot) => positions.map(frame => frame[foot].y).sort((a, b) => a-b)[Math.floor(positions.length*.3)]);
  const ratios: number[] = [];
  for (let frame = 1; frame < positions.length; frame++) {
    const dr = clip.rootAt(frame, new THREE.Vector3()).sub(clip.rootAt(frame-1, new THREE.Vector3()));
    dr.y = 0;
    if (dr.lengthSq() < .0001) continue;
    nodes.forEach((_, foot) => {
      if (Math.max(positions[frame][foot].y, positions[frame-1][foot].y) > floors[foot]+.018) return;
      const df = positions[frame][foot].clone().sub(positions[frame-1][foot]); df.y = 0;
      const ratio = -df.dot(dr)/dr.lengthSq();
      if (ratio > .25 && ratio < 1.5) ratios.push(ratio);
    });
  }
  if (ratios.length < 12) return null;
  ratios.sort((a, b) => a-b);
  return ratios[Math.floor(ratios.length/2)];
}

export class LocomotionLibrary {
  private memory = new VariantMemory();
  private actionUses = new Map<string, number>();
  lastQuality: SupportQuality | null = null;
  private constructor(readonly assets: Asset[], readonly strideScale: number, private vrm: VRM, readonly table: MotionTableData) {}

  get calibration(): { name: string; kind: Kind; scale: number; seconds: number }[] {
    return this.assets.map(asset => ({ name: asset.motion.name, kind: asset.motion.kind, scale: asset.scale, seconds: asset.clip.frames/20 }));
  }

  static async load(vrm: VRM): Promise<LocomotionLibrary> {
    const response = await fetch('/locomotion/bank.json');
    if (!response.ok) throw new Error(`基础动画读取失败 HTTP ${response.status}`);
    const bank = await response.json() as { version: number; clips: Preset[] };
    if (bank.version !== 1 || !bank.clips?.length) throw new Error('基础动画库格式无效');
    const assets = bank.clips.map(motion => ({ motion, clip: SmplClip.from(motion), scale: 1, search: new MotionSearch(motion) }));
    const walk = assets.find(asset => asset.motion.kind === 'walk');
    if (!walk) throw new Error('基础动画库缺少行走');
    const scale = calibrateStride(vrm, walk.clip);
    if (scale === null) throw new Error('无法从支撑帧校准动画步幅；不会启用未校准的移动');
    for (const asset of assets) asset.scale = calibrateStride(vrm, asset.clip)
      ?? scale*(walk.motion.sourceHeight ?? .9)/(asset.motion.sourceHeight ?? .9);
    const table = await MotionTableData.load(clip => calibrateStride(vrm, clip),
      height => scale*(walk.motion.sourceHeight ?? .9)/height);
    const reference = assets.find(asset => asset.motion.kind === 'idle')!.motion;
    const standing = { ...reference, frames: 1, localRot: reference.localRot.slice(0, 81), rootDelta: [0, 0, 0],
      footContacts: reference.footContacts?.slice(0, 4) };
    const floor = measureSupport(vrm, standing, [], 0, true).floor;
    const qualities = new Map<string, SupportQuality>();
    for (const asset of table.assets) {
      let quality = qualities.get(asset.entry.id);
      if (!quality) {
        const motion = { ...asset.clip.motion, rootDelta: asset.clip.motion.rootDelta.map(value => value*asset.scale) };
        quality = measureSupport(vrm, motion, [], 1, true); qualities.set(asset.entry.id, quality);
      }
      asset.avatarQuality = quality; asset.verticalOffset = floor-quality.floor;
    }
    return new LocomotionLibrary(assets, scale, vrm, table);
  }

  compile(plan: LocomotionPlan, seed: MotionSeed, safe: SafeRoot): PresetResult {
    if (!plan.timeline.every(step => PRESET_ACTIONS.has(step.action))) throw new Error('此动作需要 ARDY 按需生成');
    this.tableStatus = '';
    const path = plan.timeline[0]?.path?.length ? plan.timeline[0].path : plan.path;
    if (plan.timing !== 'strict' && plan.timeline.length === 1 && path.length > 2
      && ['walk', 'forward'].includes(plan.timeline[0].action) && !plan.timeline[0].fixedHeading) {
      try { return this.compileTable(plan.id, path, seed, safe); }
      catch (error) {
        if (!(error instanceof TableCoverageError)) throw error;
        this.tableStatus = error.message;
      }
    }
    let available = this.assets.filter(validStop);
    let collision: PresetCollisionError | null = null;
    let best: { result: PresetResult; quality: SupportQuality; cost: number } | null = null;
    for (let attempt = 0; attempt < 6; attempt++) {
      let candidate: { result: PresetResult; goalError: number };
      try { candidate = this.compileCandidate(plan, seed, safe, available); }
      catch (error) {
        if (error instanceof PresetUnavailableError) {
          if (best) return this.accept(best);
          if (collision) throw collision;
          throw error;
        }
        if (!(error instanceof PresetCollisionError)) throw error;
        collision = error;
        const rejected = available.find(asset => asset.motion.name === error.assetName);
        available = available.filter(asset => asset.motion.name !== error.assetName);
        if (!rejected || !available.some(asset => asset.motion.kind === rejected.motion.kind
          && (asset.motion.kind !== 'turn' || asset.motion.angle*rejected.motion.angle > 0))) {
          if (best) return this.accept(best);
          throw error;
        }
        continue;
      }
      const { result, goalError } = candidate;
      this.lastQuality = measureSupport(this.vrm, result.motion, result.phases, seed.motion.frames);
      if (goalError <= .12 && this.lastQuality.samples >= 4 && this.lastQuality.mean <= .16 && this.lastQuality.p95 <= .55) {
        const cost = this.lastQuality.mean+.25*this.lastQuality.p95+.2*goalError;
        if (!best || cost < best.cost) best = { result, quality: this.lastQuality, cost };
        if (this.lastQuality.mean <= .09 && this.lastQuality.p95 <= .32) return this.accept(best);
      }
      const used = this.assets.filter(asset => result.phases.some(phase => phase.variant === asset.motion.name));
      const families = new Set(used.flatMap(asset => asset.motion.family ? [asset.motion.family] : []));
      available = available.filter(asset => asset.motion.family ? !families.has(asset.motion.family) : !used.includes(asset));
      if (!['idle', 'start', 'walk', 'stop'].every(kind => available.some(asset => asset.motion.kind === kind))) break;
    }
    if (best) return this.accept(best);
    if (collision) throw collision;
    throw new PresetUnavailableError('当前姿态下预设组合未通过脚步质量校验，需要按需生成');
  }

  tableStatus = '';

  compileTable(id: string, path: number[][], seed: MotionSeed, safe: SafeRoot): PresetResult {
    try { return this.accept(compileMotionTable(this.vrm, this.table, id, path, seed, safe)); }
    catch (error) {
      if (error instanceof TableCollisionError) throw new PresetCollisionError(error.message, error.assetName);
      throw error;
    }
  }

  tableAction(prompt: string, seed: MotionSeed, safe: SafeRoot): PresetResult {
    return compileTableAction(this.vrm, this.table, prompt, seed, safe, this.actionUses);
  }

  private accept(candidate: { result: PresetResult; quality: SupportQuality }): PresetResult {
    this.lastQuality = candidate.quality;
    this.memory.commit(candidate.result.phases.flatMap(phase => phase.variant ? [phase.variant] : []));
    return candidate.result;
  }

  private compileCandidate(plan: LocomotionPlan, seed: MotionSeed, safe: SafeRoot, assets: Asset[]): { result: PresetResult; goalError: number } {
    const builder = new MotionBuilder(assets, seed.motion, safe, this.memory, seed.phase);
    let previousEnd = 0;
    for (const step of plan.timeline) {
      if (plan.timing === 'strict' && step.start > previousEnd) builder.hold(step.start-previousEnd);
      const before = builder.seconds;
      builder.step(step, plan.path);
      const actual = builder.seconds-before;
      if (plan.timing === 'strict') {
        if (actual > step.end-step.start+.05) throw new Error(`时间不足：${step.action} 的自然低速动画需要 ${actual.toFixed(2)} 秒；不会加速拖动`);
        builder.hold(step.end-step.start-actual);
      }
      previousEnd = step.end;
    }
    const result = builder.result(plan.id, seed.time);
    return { result, goalError: builder.goalError };
  }

  stop(seed: MotionSeed, safe: SafeRoot): PresetResult {
    const builder = new MotionBuilder(this.assets, seed.motion, safe, this.memory);
    builder.stop();
    return builder.result('stop', seed.time);
  }

  idle(position: THREE.Vector3, heading: number): MotionSeed {
    const idle = this.assets.find(asset => asset.motion.kind === 'idle');
    if (!idle) throw new Error('缺少站立动画');
    const localRot: number[] = [];
    for (let joint = 0; joint < idle.clip.joints; joint++) {
      const rotation = rotationAt(idle.clip, 0, joint);
      if (joint === 0) rotation.premultiply(new THREE.Quaternion().setFromAxisAngle(UP, heading));
      writeRotation(localRot, rotation);
    }
    return { time: 0, motion: { ...idle.motion, frames: 1, duration: .05, localRot,
      rootDelta: position.toArray(), footContacts: idle.motion.footContacts?.slice(0, 4) } };
  }
}

class MotionBuilder {
  goalError = 0;
  private rotations: number[];
  private roots: number[];
  private contacts: number[];
  private phases: MotionPhase[] = [];
  private position: THREE.Vector3;
  private heading: number;

  constructor(private assets: Asset[], private prefix: SmplMotionJson, private safe: SafeRoot, private memory: VariantMemory,
    private entryPhase?: MotionPhase['kind']) {
    this.rotations = [...prefix.localRot]; this.roots = [...prefix.rootDelta];
    this.contacts = prefix.footContacts ? [...prefix.footContacts] : new Array<number>(prefix.frames*4).fill(0);
    this.position = new THREE.Vector3().fromArray(this.roots, this.roots.length-3);
    this.heading = yawOf(rotationAt(SmplClip.from(prefix), prefix.frames-1, 0));
  }

  get seconds(): number { return this.roots.length/60; }

  hold(seconds: number): void { if (seconds > .01) this.wait(seconds); }

  step(step: LocomotionStep, fallbackPath: number[][]): void {
    const startFrame = this.roots.length/3;
    if (step.action === 'observe') {
      const before = this.seconds;
      if (this.speed() > .025) this.stop();
      this.hold(step.end-step.start-(this.seconds-before)); return;
    }
    const path = step.path?.length ? step.path : fallbackPath;
    const waypoints = path.slice(1);
    for (const [index, waypoint] of waypoints.entries()) {
      const target = new THREE.Vector3(waypoint[0], this.position.y, waypoint[2]);
      let delta = target.clone().sub(this.position);
      if (delta.length() < .09) continue;
      const desired = Math.atan2(delta.x, delta.z);
      if (step.fixedHeading && step.heading !== undefined && Math.abs(wrapped(step.heading-desired)) > .3) {
        throw new Error('当前预设库缺少侧步／倒走；固定面向与移动方向不一致时拒绝拖动角色');
      }
      if (Math.abs(wrapped(desired-this.heading)) > .3) this.turn(desired);
      delta = target.clone().sub(this.position); delta.y = 0;
      this.walk(delta.length(), Math.atan2(delta.x, delta.z), step.continuous === true || waypoints.length > 1,
        index === waypoints.length-1);
      this.goalError = Math.max(this.goalError, Math.hypot(target.x-this.position.x, target.z-this.position.z));
    }
    if (step.heading !== undefined && (step.fixedHeading || !path.length || step.distance < .01)) this.turn(step.heading);
    if (this.roots.length/3 === startFrame) this.wait(Math.min(1, step.end-step.start));
  }

  private asset(kind: Kind, distance?: number): Asset {
    const candidates = this.assets.filter(asset => asset.motion.kind === kind && validStop(asset));
    if (!candidates.length) throw new Error(`动画库缺少 ${kind}`);
    const query = this.searchBoundary();
    return this.memory.choose(candidates, asset => asset.motion.name, asset => {
      const begin = kind === 'stop' ? Math.min(15, asset.clip.frames-10) : 0;
      const end = kind === 'walk' ? Math.min(28, asset.clip.frames-4) : begin;
      const match = this.entryMatch(asset, begin, end, query);
      const minimum = this.displacement(asset, match.frame, Math.min(asset.clip.frames-1, match.frame+8));
      const goalCost = distance === undefined ? 0 : 4*Math.max(0, minimum-distance)**2;
      return match.cost+goalCost;
    });
  }

  private pairedAsset(kind: Kind, reference: Asset): Asset {
    const candidates = this.assets.filter(asset => asset.motion.kind === kind && asset.motion.family === reference.motion.family && validStop(asset));
    return candidates.length ? candidates[0] : this.asset(kind);
  }

  private poseCost(asset: Asset, frame: number, reference = this.tailRotations()): number {
    const heading = yawOf(rotationAt(asset.clip, frame, 0));
    const yaw = new THREE.Quaternion().setFromAxisAngle(UP, this.heading-heading);
    let cost = 0;
    for (let joint = 0; joint < asset.clip.joints; joint++) {
      const q = rotationAt(asset.clip, frame, joint);
      if (joint === 0) q.premultiply(yaw);
      cost += q.angleTo(reference[joint])**2*(LEG_JOINTS.includes(joint) ? 3 : 1);
    }
    return cost;
  }

  private tailRotations(): THREE.Quaternion[] {
    const count = this.prefix.bones.length, offset = this.rotations.length-count*3;
    return this.prefix.bones.map((_, joint) => {
      const vector = new THREE.Vector3().fromArray(this.rotations, offset+joint*3);
      const angle = vector.length();
      return angle < 1e-8 ? new THREE.Quaternion() : new THREE.Quaternion().setFromAxisAngle(vector.divideScalar(angle), angle);
    });
  }

  private displacement(asset: Asset, begin: number, end: number, steer = false): number {
    if (steer) return this.steeredDistance(asset, begin, end);
    const delta = asset.clip.rootAt(end, new THREE.Vector3()).sub(asset.clip.rootAt(begin, new THREE.Vector3()));
    const forward = asset.clip.rootAt(asset.clip.frames-1, new THREE.Vector3());
    const direction = asset.motion.kind === 'stop' ? 0 : Math.atan2(forward.x, forward.z);
    return Math.max(0, delta.x*Math.sin(direction)+delta.z*Math.cos(direction))*asset.scale;
  }

  private steeredDistance(asset: Asset, begin: number, end: number): number {
    const yaw = this.sourceHeading(asset, begin, end, false);
    const correction = wrapped(yaw-yawOf(rotationAt(asset.clip, end, 0)));
    const count = Math.max(2, Math.ceil(end-begin));
    let previous = asset.clip.rootAt(begin, new THREE.Vector3()), distance = 0;
    for (let index = 1; index <= count; index++) {
      const next = asset.clip.rootAt(THREE.MathUtils.lerp(begin, end, index/count), new THREE.Vector3());
      const step = next.clone().sub(previous).applyAxisAngle(UP, correction*smooth((index-.5)/count)-yaw);
      distance += step.z*asset.scale;
      previous = next;
    }
    return Math.max(0, distance);
  }

  private sourceSpeed(asset: Asset, frame: number): number {
    const before = asset.clip.rootAt(Math.max(0, frame-1), new THREE.Vector3());
    const after = asset.clip.rootAt(Math.min(asset.clip.frames-1, frame+1), new THREE.Vector3());
    return Math.hypot(after.x-before.x, after.z-before.z)*asset.scale*10;
  }

  private shortWalk(distance: number): Part[] | null {
    const moving = this.walking();
    const start = this.asset(moving ? 'walk' : 'start'), stop = this.pairedAsset('stop', start);
    const first = moving ? this.matchEntry(start, 0, 28) : 0;
    let best: { score: number; parts: Part[] } | null = null;
    for (let end = first+8; end < Math.min(start.clip.frames, first+65); end++) {
      const reference = start.motion.bones.map((_, joint) => rotationAt(start.clip, end, joint));
      reference[0].premultiply(new THREE.Quaternion().setFromAxisAngle(UP, this.heading));
      for (let begin = 0; begin < stop.clip.frames-8; begin++) {
        const total = this.displacement(start, first, end)+this.displacement(stop, begin, stop.clip.frames-1);
        const error = Math.abs(distance-total);
        if (error > .10) continue;
        const scale = 1;
        const velocityError = this.sourceSpeed(start, end)-this.sourceSpeed(stop, begin);
        const score = this.poseCost(stop, begin, reference)+error*5+velocityError*velocityError*4;
        if (!best || score < best.score) best = { score, parts: [
          { asset: start, begin: first, end, scale }, { asset: stop, begin, end: stop.clip.frames-1, scale }] };
      }
    }
    return best?.parts ?? null;
  }

  private walk(distance: number, heading: number, continuous = false, finish = true): void {
    if (distance < .08) return;
    const short = finish && !continuous && distance < 1.7 ? this.shortWalk(distance) : null;
    if (short) { for (const part of short) this.append(part, heading); return; }
    const walk = this.asset('walk', distance), start = this.pairedAsset('start', walk), stop = this.pairedAsset('stop', walk);
    const stopBegin = Math.min(18, stop.clip.frames-10);
    const stopDistance = this.displacement(stop, stopBegin, stop.clip.frames-1);
    let remaining = distance-(finish ? stopDistance : 0);
    if (remaining <= .08) throw new Error('移动距离不足一个安全落脚步；请合并短步命令');
    if (!this.walking() && remaining > this.displacement(start, 0, start.clip.frames-1, continuous)+.1) {
      this.append({ asset: start, begin: 0, end: start.clip.frames-1, scale: 1, steer: continuous }, heading);
      remaining -= this.displacement(start, 0, start.clip.frames-1, continuous);
    }
    while (remaining > .03) {
      const begin = this.matchEntry(walk, 0, Math.min(28, walk.clip.frames-4));
      let end = begin+1;
      while (end < walk.clip.frames-1 && this.displacement(walk, begin, end, continuous) < remaining) end++;
      const lastLength = this.displacement(walk, begin, end-1, continuous);
      let length = this.displacement(walk, begin, end, continuous);
      if (length < .005) throw new Error('行走预设缺少有效根运动');
      if (length > remaining) {
        end = end-1+(remaining-lastLength)/Math.max(.00001, length-lastLength);
        length = this.displacement(walk, begin, end, continuous);
      }
      this.append({ asset: walk, begin, end, scale: 1, steer: continuous }, heading);
      remaining -= length;
    }
    if (finish) this.append({ asset: stop, begin: stopBegin, end: stop.clip.frames-1, scale: 1 }, heading);
  }

  private matchEntry(asset: Asset, begin: number, end: number): number {
    return this.entryMatch(asset, begin, end, this.searchBoundary()).frame;
  }

  private searchBoundary(): number[] {
    const motion = { ...this.prefix, frames: this.roots.length/3, localRot: this.rotations,
      rootDelta: this.roots, footContacts: this.contacts };
    return boundaryFeatures(motion);
  }

  private entryMatch(asset: Asset, begin: number, end: number, query: number[]): { frame: number; cost: number } {
    const rate = asset.motion.kind === 'idle' ? 1 : Math.min(1, .5/Math.max(.25,
      asset.motion.distance*asset.scale/asset.motion.duration));
    const candidates = asset.search.shortlist(query, begin, end, rate, asset.scale);
    let chosen = begin, best = Infinity;
    const reference = this.tailRotations();
    for (const candidate of candidates) {
      const cost = this.poseCost(asset, candidate.frame, reference)/asset.clip.joints+candidate.cost;
      if (cost < best) { best = cost; chosen = candidate.frame; }
    }
    return { frame: chosen, cost: best };
  }

  private turn(target: number): void {
    if (this.speed() > .1 && (this.phases.at(-1)?.kind ?? this.entryPhase) !== 'turn') this.stop();
    const angle = wrapped(target-this.heading);
    if (Math.abs(angle) < .025) return;
    if (Math.abs(angle) > THREE.MathUtils.degToRad(100)) {
      this.turn(this.heading+angle/2); this.turn(target); return;
    }
    const candidates = this.assets.filter(asset => asset.motion.kind === 'turn' && asset.motion.angle*angle > 0);
    candidates.sort((a, b) => Math.abs(a.motion.angle-angle)-Math.abs(b.motion.angle-angle));
    if (!candidates.length) throw new PresetUnavailableError('缺少对应方向的迈步转身动画');
    const nearest = Math.abs(candidates[0].motion.angle-angle);
    const compatible = candidates.filter(asset => Math.abs(asset.motion.angle-angle) <= nearest+.15);
    const asset = this.memory.choose(compatible, value => value.motion.name,
      value => this.poseCost(value, 0)/value.clip.joints+Math.abs(value.motion.angle-angle)*2);
    this.append({ asset, begin: 0, end: asset.clip.frames-1, scale: 1 }, this.heading, angle);
  }

  stop(): void {
    if (this.speed() > .025) {
      const stop = this.asset('stop'), begin = this.matchEntry(stop, 15, stop.clip.frames-10);
      this.append({ asset: stop, begin, end: stop.clip.frames-1, scale: 1 }, this.heading);
    } else this.settle();
  }

  private settle(): void {
    const stop = this.asset('stop');
    const begin = stop.clip.frames-9, end = stop.clip.frames-1;
    const yaw = wrapped(yawOf(rotationAt(stop.clip, end, 0))-yawOf(rotationAt(stop.clip, begin, 0)));
    const alignment = new THREE.Quaternion().setFromAxisAngle(UP,
      this.heading-yaw-yawOf(rotationAt(stop.clip, begin, 0)));
    const residual = this.poseResidual(stop.clip, begin, alignment);
    const maximum = Math.max(...residual.map(rotation => rotation.angleTo(new THREE.Quaternion())));
    const blendFrames = Math.max(8, Math.ceil(maximum/THREE.MathUtils.degToRad(7.5)));
    this.append({ asset: stop, begin, end, scale: 1 }, this.heading-yaw, undefined, blendFrames);
  }

  private wait(seconds: number): void {
    let remaining = Math.max(1, Math.ceil(seconds*20));
    while (remaining > 0) {
      const asset = this.asset('idle');
      const count = Math.min(Math.ceil(remaining/2), asset.clip.frames-1);
      const heading = this.heading;
      this.append({ asset, begin: 0, end: count, scale: 1 }, heading, undefined, count);
      this.append({ asset, begin: count, end: 0, scale: 1 }, this.heading, undefined, count);
      remaining -= count*2;
    }
  }

  private speed(): number {
    if (this.roots.length < 6) return 0;
    return Math.hypot(this.roots.at(-3)!-this.roots.at(-6)!, this.roots.at(-1)!-this.roots.at(-4)!)*20;
  }

  private walking(): boolean {
    const phase = this.phases.at(-1)?.kind ?? this.entryPhase;
    return phase !== 'turn' && phase !== 'stop' && phase !== 'idle' && this.speed() > .08;
  }

  private append(part: Part, heading: number, turn?: number, blendFrames = 8): void {
    const { asset, begin, end } = part, clip = asset.clip;
    const sourceYaw = this.sourceHeading(asset, begin, end, turn !== undefined);
    const alignment = new THREE.Quaternion().setFromAxisAngle(UP, heading-sourceYaw);
    const nativeYaw = asset.motion.kind === 'turn' ? asset.motion.angle
      : wrapped(yawOf(rotationAt(clip, end, 0))-yawOf(rotationAt(clip, begin, 0)));
    // Carry the native root steps with the same gradual yaw as the pose. Clip
    // heading drift must not accumulate into a spurious stop/turn at each seam.
    const correction = turn !== undefined ? turn-nativeYaw
      : part.steer
        ? wrapped(sourceYaw-yawOf(rotationAt(clip, end, 0))) : 0;
    const distance = this.displacement(asset, begin, end)*part.scale;
    const rate = asset.motion.kind === 'idle' ? 1 : asset.motion.kind === 'turn' ? this.turnRate(asset, turn ?? nativeYaw)
      : THREE.MathUtils.clamp(.5/Math.max(.25, distance/((end-begin)/20)), .45, 1);
    const count = Math.max(2, blendFrames, Math.ceil(Math.abs(end-begin)/rate));
    const residual = this.poseResidual(clip, begin, alignment);
    let previous = clip.rootAt(begin, new THREE.Vector3());
    const heightResidual = this.position.y-(previous.y+(asset.motion.heightOffset ?? 0))*asset.scale;
    const first = this.roots.length/3;
    for (let index = 1; index <= count; index++) {
      const t = index/count, frame = THREE.MathUtils.lerp(begin, end, asset.motion.kind === 'idle' ? smooth(t) : t);
      const source = clip.rootAt(frame, new THREE.Vector3());
      const step = source.clone().sub(previous).multiplyScalar(asset.scale*part.scale);
      const yaw = new THREE.Quaternion().setFromAxisAngle(UP, correction*smooth((index-.5)/count));
      const next = step.applyQuaternion(yaw).applyQuaternion(alignment).add(this.position);
      next.y = (source.y+(asset.motion.heightOffset ?? 0))*asset.scale
        + heightResidual*(1-smooth(Math.min(1, index/blendFrames)));
      const collision = this.safe(this.position, next);
      if (collision) throw new PresetCollisionError(`预设根运动无法安全通过：${collision}；已拒绝该片段`, asset.motion.name);
      this.emitPose(clip, frame, alignment, correction*smooth(t), residual, index, blendFrames);
      this.position.copy(next); this.roots.push(...next.toArray()); previous = source;
      const offset = Math.round(frame)*4;
      this.contacts.push(...(asset.motion.footContacts?.slice(offset, offset+4) ?? [0, 0, 0, 0]));
    }
    this.heading = yawOf(this.tailRotations()[0]);
    this.phases.push({ kind: asset.motion.kind, start: first/20, end: this.roots.length/60,
      source: asset.motion.source, variant: asset.motion.name });
  }

  private sourceHeading(asset: Asset, begin: number, end: number, turning: boolean): number {
    if (turning || asset.motion.kind === 'idle' || end-begin <= 10) return yawOf(rotationAt(asset.clip, begin, 0));
    if (asset.motion.kind === 'stop') return 0;
    const travel = asset.clip.rootAt(asset.clip.frames-1, new THREE.Vector3());
    return Math.atan2(travel.x, travel.z);
  }

  private turnRate(asset: Asset, target: number): number {
    let maximum = .01;
    for (let frame = 1; frame < asset.clip.frames; frame++) {
      maximum = Math.max(maximum, rotationAt(asset.clip, frame, 0).angleTo(rotationAt(asset.clip, frame-1, 0)));
    }
    const warp = Math.max(1, Math.abs(target/asset.motion.angle));
    return Math.min(1.15, THREE.MathUtils.degToRad(70)/(maximum*20*warp));
  }

  private poseResidual(clip: SmplClip, frame: number, alignment: THREE.Quaternion): THREE.Quaternion[] {
    return this.tailRotations().map((last, joint) => {
      const target = rotationAt(clip, frame, joint);
      if (joint === 0) target.premultiply(alignment);
      return last.multiply(target.invert());
    });
  }

  private emitPose(clip: SmplClip, frame: number, alignment: THREE.Quaternion,
    yaw: number, residual: THREE.Quaternion[], elapsed: number, blendFrames: number): void {
    const weight = 1-smooth(Math.min(1, elapsed/blendFrames));
    const rootYaw = new THREE.Quaternion().setFromAxisAngle(UP, yaw).premultiply(alignment);
    for (let joint = 0; joint < clip.joints; joint++) {
      const target = rotationAt(clip, frame, joint);
      if (joint === 0) target.premultiply(rootYaw);
      const offset = new THREE.Quaternion().slerp(residual[joint], weight);
      writeRotation(this.rotations, target.premultiply(offset));
    }
  }

  result(name: string, time: number): PresetResult {
    const frames = this.roots.length/3;
    return { motion: { ...this.prefix, name: `preset_${name}`, source: 'local-amass-root-motion',
      dataset: '预设低速 / 原生转身', frames, duration: frames/20,
      localRot: this.rotations, rootDelta: this.roots, footContacts: this.contacts },
      phases: this.phases, seconds: Math.max(0, (frames-1)/20-time) };
  }
}

/** Preserve a short contact-aware lead; repeated commands never reset the gait. */
export function presetSeed(clip: SmplClip, time: number, playing: boolean, offset: THREE.Vector3,
  retainTail = false, lead?: SeedLead): MotionSeed {
  const current = Math.max(0, Math.min(clip.frames-1, Math.floor(time*clip.fps+1e-6)));
  let end = current+1;
  if (playing) {
    const minimum = lead ? Math.max(1, Math.ceil(lead.minimum*clip.fps)) : 2;
    const maximum = lead ? Math.max(minimum, Math.floor(lead.maximum*clip.fps)) : 5;
    const limit = Math.min(clip.frames-1, current+maximum);
    end = limit+1;
    for (let frame = current+minimum; frame <= limit; frame++) {
      const contacts = clip.motion.footContacts?.slice(frame*4, frame*4+4);
      if (contacts?.slice(0, 2).some(Boolean) && contacts.slice(2).some(Boolean)) { end = frame+1; break; }
    }
  }
  if (retainTail) end = clip.frames;
  const start = Math.max(0, current-40), rootDelta = clip.motion.rootDelta.slice(start*3, end*3);
  for (let index = 0; index < rootDelta.length; index += 3) {
    rootDelta[index] += offset.x; rootDelta[index+1] += offset.y; rootDelta[index+2] += offset.z;
  }
  return { time: Math.max(0, Math.min(time, (clip.frames-1)/clip.fps)-start/clip.fps),
    motion: { ...clip.motion, frames: end-start, duration: (end-start)/clip.fps,
    localRot: clip.motion.localRot.slice(start*81, end*81), rootDelta,
    footContacts: clip.motion.footContacts?.slice(start*4, end*4) } };
}

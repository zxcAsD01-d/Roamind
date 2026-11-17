import { Quaternion, Vector3 } from 'three';
import { dampedStep } from '../vendor/HAST-SEC/motion/v9/damped_output.mjs';
import { SmplClip } from './retarget';
import type { MotionPhase, MotionSeed, PresetResult } from './locomotion';
import { tableRotation, tableYaw, rotationVector, vectorRotation, yawRotation } from './motion_table_features';
import { TableCoverageError, transitionKey, type TablePlacement, type TableRoute } from './motion_table_search';

type Residual = { offset: Vector3; velocity: Vector3 };
export type TableSafeRoot = (previous: Vector3, next: Vector3) => string | null;
export class TableTransitionError extends TableCoverageError {
  constructor(message: string, readonly key: string) { super(message); }
}
export class TableCollisionError extends Error {
  constructor(message: string, readonly assetName: string) { super(message); }
}
function advance(residual: Residual, hz: number): Vector3 {
  const response = dampedStep(residual.offset.clone().negate(), residual.velocity, hz, .05);
  residual.offset.add(response.move); residual.velocity.copy(response.velocity);
  return residual.offset;
}
function boundaryResiduals(prefix: SmplClip, part: TablePlacement, alignment: Quaternion): Residual[] {
  const { clip, begin, rate } = part.asset, last = prefix.frames-1;
  return prefix.motion.bones.map((_, joint) => {
    const previous = tableRotation(prefix, Math.max(0, last-1), joint), old = tableRotation(prefix, last, joint);
    const target = tableRotation(clip, begin, joint), following = tableRotation(clip, begin+rate, joint);
    if (!joint) { target.premultiply(alignment); following.premultiply(alignment); }
    const outgoing = rotationVector(previous.invert().multiply(old)).multiplyScalar(20);
    const incoming = rotationVector(target.clone().invert().multiply(following)).multiplyScalar(20);
    return { offset: rotationVector(target.invert().multiply(old)), velocity: outgoing.sub(incoming) };
  });
}
class TableBuilder {
  private rotations: number[];
  private roots: number[];
  private contacts: number[];
  private phases: MotionPhase[] = [];
  constructor(private seed: MotionSeed, private safe: TableSafeRoot) {
    this.rotations = [...seed.motion.localRot]; this.roots = [...seed.motion.rootDelta];
    this.contacts = seed.motion.footContacts ? [...seed.motion.footContacts] : new Array<number>(seed.motion.frames*4).fill(0);
  }
  private current(): SmplClip {
    return SmplClip.from({ ...this.seed.motion, frames: this.roots.length/3, localRot: this.rotations,
      rootDelta: this.roots, footContacts: this.contacts });
  }
  append(part: TablePlacement): void {
    const { clip, begin, end, scale, rate, entry } = part.asset;
    const alignment = yawRotation(part.heading-tableYaw(tableRotation(clip, begin, 0)));
    const prefix = this.current(), residuals = boundaryResiduals(prefix, part, alignment);
    const origin = clip.rootAt(begin, new Vector3()), start = prefix.rootAt(prefix.frames-1, new Vector3());
    const vertical = part.asset.verticalOffset ?? entry.heightOffset*scale;
    const sourceY = origin.y*scale+vertical;
    const height: Residual = { offset: new Vector3(0, start.y-sourceY, 0), velocity: new Vector3() };
    const count = Math.ceil((end-begin)/rate), first = this.roots.length/3;
    let previous = start;
    for (let i = 1; i <= count; i++) {
      const frame = begin+(end-begin)*i/count;
      const source = clip.rootAt(frame, new Vector3()), next = source.clone().sub(origin).multiplyScalar(scale).applyQuaternion(alignment).add(start);
      next.y = source.y*scale+vertical+advance(height, 2.2).y;
      const collision = this.safe(previous, next);
      if (collision) throw new TableCollisionError(`动作表原生根轨迹碰撞：${collision}`, entry.name);
      this.emit(clip, frame, alignment, residuals);
      this.roots.push(...next.toArray()); previous = next;
      this.contacts.push(...(clip.motion.footContacts?.slice(Math.round(frame)*4, Math.round(frame)*4+4) ?? [0, 0, 0, 0]));
    }
    this.phases.push({ kind: entry.role === 'travel' ? 'walk' : entry.role,
      start: first/20, end: this.roots.length/60, source: entry.source, variant: entry.id });
  }
  private emit(clip: SmplClip, frame: number, alignment: Quaternion, residuals: Residual[]): void {
    for (let joint = 0; joint < 27; joint++) {
      const target = tableRotation(clip, frame, joint);
      if (!joint) target.premultiply(alignment);
      // Only the seam residual decays. Native gait is never low-pass filtered.
      const correction = vectorRotation(advance(residuals[joint], 2.2));
      this.rotations.push(...rotationVector(target.multiply(correction).normalize()).toArray());
    }
  }
  result(id: string, route: TableRoute): PresetResult {
    const frames = this.roots.length/3;
    const motion = { ...this.seed.motion, name: 'table_'+id, source: 'native-motion-table', dataset: '全身动作表 / Lumen 接缝',
      frames, duration: frames/20, localRot: this.rotations, rootDelta: this.roots, footContacts: this.contacts };
    const clip = SmplClip.from(motion);
    for (let f = this.seed.motion.frames; f < frames; f++) for (let j = 0; j < 27; j++) {
      if (tableRotation(clip, f-1, j).angleTo(tableRotation(clip, f, j)) > .4) {
        const index = this.phases.findIndex(phase => f/20 >= phase.start && f/20 < phase.end);
        const part = route.parts[Math.max(0, index)], previous = index > 0 ? route.parts[index-1].asset : null;
        throw new TableTransitionError(`动作表接缝关节变化过大：帧 ${f} 关节 ${j}`, transitionKey(previous, part.asset));
      }
    }
    return { motion, phases: this.phases, seconds: Math.max(0, (frames-1)/20-this.seed.time), table: route.diagnostics };
  }
}
export function assembleMotionTable(id: string, route: TableRoute, seed: MotionSeed, safe: TableSafeRoot): PresetResult {
  const builder = new TableBuilder(seed, safe);
  for (const part of route.parts) builder.append(part);
  return builder.result(id, route);
}

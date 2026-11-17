import { Vector3 } from 'three';
import { SmplClip } from './retarget';
import type { MotionSeed } from './locomotion';
import type { TableAsset } from './motion_table_data';
import { tableBoundary, tableCost, tableRotation, tableYaw, wrapYaw } from './motion_table_features';
import type { TableDiagnostics } from './motion_table_types';

export class TableCoverageError extends Error {}
export type TablePlacement = { asset: TableAsset; x: number; z: number; heading: number };
export type TableRoute = { parts: TablePlacement[]; diagnostics: TableDiagnostics; alternatives?: TableRoute[] };
type Point = { x: number; z: number; s: number };
type State = { x: number; z: number; heading: number; progress: number; cost: number; maximum: number;
  feature: number[]; parts: TablePlacement[]; last: TableAsset | null };
export const transitionKey = (previous: TableAsset | null, next: TableAsset): string =>
  `${previous ? previous.entry.id+'@'+previous.begin : 'seed'}>${next.entry.id}@${next.begin}`;
const SEAM_LIMITS = { rms: .45, maximum: 1.2, velocity: .18, cost: .65 };

class Route {
  readonly points: Point[] = [];
  constructor(path: number[][], origin: Vector3) {
    let previous = { x: origin.x, z: origin.z, s: 0 }; this.points.push(previous);
    for (const row of path) {
      if (row.length < 3 || !row.every(Number.isFinite)) throw new Error('路线包含无效坐标');
      const distance = Math.hypot(row[0]-previous.x, row[2]-previous.z);
      if (distance < .01) continue;
      previous = { x: row[0], z: row[2], s: previous.s+distance }; this.points.push(previous);
    }
    if (this.points.length > 512 || this.length > 30) throw new TableCoverageError('单次检索路线超过预算，请分为连续路段');
  }
  get length(): number { return this.points.at(-1)!.s; }
  at(s: number): Point {
    const value = Math.max(0, Math.min(this.length, s));
    let low = 1, high = this.points.length-1;
    while (low < high) { const mid = (low+high)>>1; if (this.points[mid].s < value) low = mid+1; else high = mid; }
    const a = this.points[low-1], b = this.points[low], t = (value-a.s)/(b.s-a.s);
    return { x: a.x+(b.x-a.x)*t, z: a.z+(b.z-a.z)*t, s: value };
  }
  project(x: number, z: number, progress: number): { s: number; distance: number } {
    let best = { s: progress, distance: Infinity };
    for (let i = 1; i < this.points.length; i++) {
      const a = this.points[i-1], b = this.points[i];
      if (b.s < progress-.08 || a.s > progress+3) continue;
      const dx = b.x-a.x, dz = b.z-a.z, t = Math.max(0, Math.min(1, ((x-a.x)*dx+(z-a.z)*dz)/(dx*dx+dz*dz)));
      const distance = Math.hypot(x-a.x-dx*t, z-a.z-dz*t);
      if (distance < best.distance) best = { s: a.s+(b.s-a.s)*t, distance };
    }
    return best;
  }
}
function transformed(state: State, asset: TableAsset, correction: number, sample = 8): Point {
  const row = asset.track[sample], yaw = state.heading+correction, c = Math.cos(yaw), s = Math.sin(yaw);
  return { x: state.x+row[0]*c+row[1]*s, z: state.z-row[0]*s+row[1]*c, s: row[3] };
}
function correctedFeature(feature: number[], correction: number): number[] {
  const output = [...feature], c = Math.cos(correction), s = Math.sin(correction);
  output[0] = feature[0]*c+feature[1]*s; output[1] = -feature[0]*s+feature[1]*c;
  return output;
}
function evaluate(route: Route, state: State, asset: TableAsset, correction: number, blocked: Set<string>): State | null {
  if (blocked.has(transitionKey(state.last, asset))) return null;
  const end = transformed(state, asset, correction), projected = route.project(end.x, end.z, state.progress);
  if (projected.s < state.progress+.04 || projected.distance > .3) return null;
  let deviation = 0, maximum = state.maximum;
  for (let i = 1; i <= 8; i++) {
    const point = transformed(state, asset, correction, i), target = route.at(state.progress+point.s);
    const error = Math.hypot(point.x-target.x, point.z-target.z);
    if (error > .4) return null;
    maximum = Math.max(maximum, error); deviation += error*error/8;
  }
  const seam = tableCost(state.feature, correctedFeature(asset.first, correction));
  if (seam.rms > SEAM_LIMITS.rms || seam.maximum > SEAM_LIMITS.maximum || seam.velocity > SEAM_LIMITS.velocity
    || seam.cost > SEAM_LIMITS.cost) return null;
  const placement = { asset, x: state.x, z: state.z, heading: state.heading+correction };
  const repeat = state.parts.slice(-3).filter(part => part.asset.entry.id === asset.entry.id).length;
  return { x: end.x, z: end.z, heading: wrapYaw(placement.heading+asset.yaw), progress: projected.s,
    feature: asset.last, parts: [...state.parts, placement], last: asset, maximum,
    cost: state.cost+deviation*asset.length*8+seam.cost*.12+correction**2*.5+repeat*.025+(asset.avatarQuality?.mean ?? 0)*asset.length*.8 };
}
function shortlist(route: Route, state: State, assets: TableAsset[], role: string): TableAsset[] {
  return assets.filter(a => a.entry.role === role && (!a.avatarQuality
    || a.avatarQuality.samples >= 4 && a.avatarQuality.mean <= .12 && a.avatarQuality.p95 <= .32)).map(asset => {
    const end = transformed(state, asset, 0), target = route.at(state.progress+asset.length);
    const seam = tableCost(state.feature, asset.first);
    return { asset, score: Math.hypot(end.x-target.x, end.z-target.z)*2+seam.cost*.5, seam };
  }).filter(row => row.seam.rms <= SEAM_LIMITS.rms && row.seam.maximum <= SEAM_LIMITS.maximum
    && row.seam.velocity < SEAM_LIMITS.velocity+.08).sort((a, b) => a.score-b.score).slice(0, 36).map(row => row.asset);
}
function diagnostics(state: State, started: number, expanded: number, goalError: number): TableDiagnostics {
  let seamPoseRms = 0, seamJointMax = 0, seamVelocityMax = 0;
  for (let i = 1; i < state.parts.length; i++) {
    const previous = state.parts[i-1], current = state.parts[i];
    const correction = wrapYaw(current.heading-previous.heading-previous.asset.yaw);
    const seam = tableCost(previous.asset.last, correctedFeature(current.asset.first, correction));
    seamPoseRms = Math.max(seamPoseRms, seam.rms); seamJointMax = Math.max(seamJointMax, seam.maximum);
    seamVelocityMax = Math.max(seamVelocityMax, seam.velocity);
  }
  return { entries: state.parts.map(part => part.asset.entry.id), searchMs: performance.now()-started, expanded,
    goalError, pathMaxError: state.maximum, seamPoseRms, seamJointMax, seamVelocityMax };
}
function prune(states: State[], length: number): State[] {
  states.sort((a, b) => (a.cost+(length-a.progress)*.3)-(b.cost+(length-b.progress)*.3));
  const cells = new Set<string>();
  return states.filter(state => {
    const key = [Math.round(state.progress/.2), Math.round(state.heading/.15), state.last?.entry.id, state.last?.begin].join(':');
    if (cells.has(key)) return false; cells.add(key); return true;
  }).slice(0, 12);
}

type Completed = { state: State; score: number; error: number; key: string };
function remember(completed: Completed[], state: State, route: Route): void {
  const goal = route.at(route.length), error = Math.hypot(state.x-goal.x, state.z-goal.z);
  if (error > .12) return;
  const score = state.cost+error*error*30;
  const key = state.parts.map(part => part.asset.entry.id+'@'+part.asset.begin).join(',');
  const existing = completed.findIndex(item => item.key === key);
  if (existing >= 0 && completed[existing].score <= score) return;
  if (existing >= 0) completed.splice(existing, 1);
  completed.push({ state, score, error, key }); completed.sort((a, b) => a.score-b.score);
  if (completed.length > 10) completed.pop();
}

export function searchMotionTable(assets: TableAsset[], path: number[][], seed: MotionSeed, blocked = new Set<string>()): TableRoute {
  const started = performance.now(), clip = SmplClip.from(seed.motion), origin = clip.rootAt(clip.frames-1, new Vector3());
  const route = new Route(path, origin);
  if (route.length < .4) throw new TableCoverageError('路线短于一个自然落脚步');
  const initial: State = { x: origin.x, z: origin.z, heading: tableYaw(tableRotation(clip, clip.frames-1, 0)),
    feature: tableBoundary(clip), progress: 0, cost: 0, maximum: 0, parts: [], last: null };
  const moving = Math.hypot(initial.feature[0], initial.feature[1]) > .08 && !['stop', 'idle', 'turn'].includes(seed.phase ?? '');
  let beam = [initial], expanded = 0;
  const completed: Completed[] = [];
  for (let depth = 0; depth < 55 && performance.now()-started < 180; depth++) {
    const next: State[] = [];
    for (const state of beam) {
      const finishing = state.progress > route.length-1.3;
      if (finishing) for (const asset of shortlist(route, state, assets, 'stop')) for (const correction of [-.12, 0, .12]) {
        expanded++; const end = evaluate(route, state, asset, correction, blocked);
        if (!end) continue;
        remember(completed, end, route);
      }
      if (state.progress > route.length-.2) continue;
      const role = !state.parts.length && !moving ? 'start' : 'travel';
      for (const asset of shortlist(route, state, assets, role)) for (const correction of [-.12, 0, .12]) {
        expanded++; const candidate = evaluate(route, state, asset, correction, blocked);
        if (candidate && (completed.length < 10 || candidate.cost < completed.at(-1)!.score)) next.push(candidate);
      }
    }
    if (!next.length) break;
    beam = prune(next, route.length);
  }
  if (!completed.length) throw new TableCoverageError(`动作表无法在原生步态与接缝门限内覆盖路线（搜索 ${expanded} 个连接）`);
  const routes = completed.map(item => ({ parts: item.state.parts, diagnostics: diagnostics(item.state, started, expanded, item.error) }));
  return { ...routes[0], alternatives: routes.slice(1) };
}

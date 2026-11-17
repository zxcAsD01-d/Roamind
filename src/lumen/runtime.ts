import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';
import { beatLayers } from '../../vendor/HAST-SEC/motion/beat_layers.mjs';
import { outputSmoothing } from '../../vendor/HAST-SEC/motion/v9/output_smoothing.mjs';
import { clipRotation } from '../../vendor/HAST-SEC/motion/skill_rotation.mjs';
import { LumenAssets } from './assets';
import type { LayerEngine, LumenAction, LumenEntry, LumenSkill, SmoothingEngine, SmoothPoint } from './types';

type Report = { id: string; status: 'playing' | 'completed' | 'failed' | 'interrupted'; detail?: string };
type Playing = { command: string; track: number; mask: number[]; action: LumenAction; elapsed: number; draining: number };
export type PreparedGesture = { action: LumenAction; entry: LumenEntry; sources: Map<string, Float32Array>; duration: number };
const isFinitePose = (p: SmoothPoint): boolean => p.p.toArray().every(Number.isFinite)
  && p.q.every(q => q.toArray().every(Number.isFinite) && Math.abs(q.lengthSq()-1) < 1e-4);
const CANONICAL_SIGN = (vrm: VRM): number => vrm.meta.metaVersion === '0' ? -1 : 1;

/** Root-motion playback stays authoritative; HAST owns independent, reversible arm tracks. */
export class LumenMotionSystem {
  private engine: LayerEngine;
  private smoothing: SmoothingEngine;
  private nodes: (THREE.Object3D | null)[];
  private arms: number[];
  private saved: { bone: THREE.Object3D; rotation: THREE.Quaternion }[] = [];
  private playing: Playing[] = [];
  private reports: Report[] = [];
  private uses = new Map<string, number>();
  private sourceMap = new Map<string, Float32Array>();
  private lastMask = new Set<number>();
  private lastError = '';

  private constructor(private vrm: VRM, private assets: LumenAssets) {
    const names = assets.manifest.bones;
    this.nodes = names.map(name => vrm.humanoid.getNormalizedBoneNode(name as VRMHumanBoneName));
    this.arms = names.flatMap((name, i) => /^(left|right)(Shoulder|UpperArm|LowerArm|Hand)$/.test(name) ? [i] : []);
    this.engine = beatLayers(names, this.sourceMap);
    this.smoothing = outputSmoothing(this.arms.map(i => names[i]), { output_smoothing: {
      arms_hz: 2, arms_max_hz: 5, hands_hz: 2, hands_max_hz: 5, body_hz: 2,
      body_max_hz: 5, arms_beta: 2, beta: 1, window_seconds: .3,
    } });
  }

  static async load(vrm: VRM): Promise<LumenMotionSystem> {
    return new LumenMotionSystem(vrm, await LumenAssets.load());
  }

  async prepare(action: LumenAction, duration: number): Promise<PreparedGesture> {
    if (!Number.isFinite(duration) || duration < 1.5 || duration > 7.5) throw new Error('流明上身动作时长应为 1.5–7.5 秒');
    const original = await this.assets.entry(action);
    const entry = { ...original, skills: original.skills.filter(skill => skill.duration/duration >= .4 && skill.duration/duration <= 2) };
    if (!entry.skills.length) throw new Error('当前动作库没有与请求时长匹配的原生片段');
    if (entry.mask.some(i => !this.nodes[i])) throw new Error('当前模型缺少动作要求的手臂骨骼');
    const rows = await Promise.all([...new Set(entry.skills.map(skill => skill.source))]
      .map(async source => [source, await this.assets.source(source)] as const));
    return { action, entry, duration, sources: new Map(rows) };
  }

  start(command: string, prepared: PreparedGesture): void {
    const { entry, action, sources, duration } = prepared;
    const base = this.canonicalPose(), skill = this.select(entry, sources, base);
    this.sourceMap.set(skill.source, sources.get(skill.source)!);
    const speed = skill.duration/duration;
    if (speed < .4 || speed > 2) throw new Error('请求时长超出流明原生动作速度范围');
    const track = this.engine.add({ ...entry, skills: [skill] }, { speed, hold: 0, cycles: 1 }, base);
    for (const old of this.playing.filter(old => old.mask.some(i => entry.mask.includes(i))))
      this.reports.push({ id: old.command, status: 'interrupted', detail: '由流明重叠轨道平滑切换' });
    this.playing = this.playing.filter(old => !old.mask.some(i => entry.mask.includes(i)));
    this.playing.push({ command, track, mask: entry.mask, action, elapsed: 0, draining: 0 });
    this.uses.set(skill.source+':'+skill.start, (this.uses.get(skill.source+':'+skill.start) ?? 0)+1);
    for (const i of entry.mask) this.lastMask.add(i);
    this.reports.push({ id: command, status: 'playing', detail: 'lumen-native-beatLayers' });
    this.lastError = '';
  }

  restore(): void {
    for (const saved of this.saved) saved.bone.quaternion.copy(saved.rotation);
    this.saved = [];
  }

  update(dt: number): void {
    if (!this.active) return;
    try { this.step(dt); }
    catch (error) {
      this.lastError = error instanceof Error ? error.message : '流明动作执行失败';
      this.cancel(this.lastError, 'failed');
      console.error('流明动作层已停止', error);
    }
  }

  private step(dt: number): void {
    const base = this.canonicalPose(), raw = this.engine.sample(base, dt);
    const point = { q: this.arms.map(i => new THREE.Quaternion().fromArray(base[i]).normalize().invert()
      .multiply(new THREE.Quaternion().fromArray(raw[i]).normalize())), p: new THREE.Vector3() };
    const value = this.smoothing.step(point, dt, null, p => ({ q: p.q, p: point.p }), isFinitePose);
    const sign = CANONICAL_SIGN(this.vrm);
    this.arms.forEach((i, slot) => {
      const bone = this.nodes[i];
      if (!bone || !this.lastMask.has(i)) return;
      this.saved.push({ bone, rotation: bone.quaternion.clone() });
      const q = new THREE.Quaternion().fromArray(base[i]).normalize().multiply(value.q[slot]);
      bone.quaternion.set(q.x*sign, q.y, q.z*sign, q.w);
    });
    this.finishTracks(dt, value);
    this.vrm.scene.updateWorldMatrix(true, true);
  }

  private finishTracks(dt: number, value: SmoothPoint): void {
    const tracks = this.engine.state().tracks;
    const active = new Set(tracks.map(track => track.id));
    const sources = new Set(tracks.map(track => track.source));
    for (const source of this.sourceMap.keys()) if (!sources.has(source)) this.sourceMap.delete(source);
    this.playing = this.playing.filter(row => {
      row.elapsed += dt;
      if (active.has(row.track)) return true;
      row.draining += dt;
      const gap = Math.max(...row.mask.map(i => value.q[this.arms.indexOf(i)].angleTo(new THREE.Quaternion())));
      if (gap > .008) {
        if (row.draining > 5) throw new Error('动作撤回未收敛，已停止而非强制完成');
        return true;
      }
      this.reports.push({ id: row.command, status: 'completed', detail: '动作轨道与撤回已结束' });
      return false;
    });
    if (!this.playing.length && !this.engine.active()) { this.lastMask.clear(); this.smoothing.reset(); }
  }

  cancel(detail = '动作已取消', status: 'interrupted' | 'failed' = 'interrupted'): void {
    this.restore();
    this.reports.push(...this.playing.map(row => ({ id: row.command, status, detail })));
    this.playing = []; this.engine.reset(); this.smoothing.reset(); this.lastMask.clear(); this.sourceMap.clear();
  }

  get active(): boolean { return this.playing.length > 0 || this.engine.active(); }
  drainReports(): Report[] { return this.reports.splice(0); }
  state(): { engine: string; active: boolean; actions: string[]; tracks: number; error: string } {
    return { engine: 'HAST-SEC/beatLayers+outputSmoothing', active: this.active,
      actions: this.playing.map(row => row.action), tracks: this.engine.state().tracks.length, error: this.lastError };
  }

  private canonicalPose(): number[][] {
    const sign = CANONICAL_SIGN(this.vrm);
    return this.nodes.map(node => {
      const q = node?.quaternion ?? new THREE.Quaternion(); return [q.x*sign, q.y, q.z*sign, q.w];
    });
  }

  private select(entry: LumenEntry, sources: Map<string, Float32Array>, base: number[][]): LumenSkill {
    const cost = (skill: LumenSkill): number => {
      const source = sources.get(skill.source);
      if (!source || skill.start < 0 || skill.end*53*4 > source.length) throw new Error('流明动作帧区间越界');
      const pose = entry.mask.reduce((sum, i) => sum+new THREE.Quaternion().fromArray(base[i]).normalize()
        .angleTo(clipRotation(source, skill.start, i, this.nodes.length, skill))**2, 0);
      return pose+.08*(this.uses.get(skill.source+':'+skill.start) ?? 0);
    };
    return [...entry.skills].sort((a, b) => cost(a)-cost(b))[0];
  }
}

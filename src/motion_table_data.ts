import { SmplClip } from './retarget';
import { TABLE_JOINTS, TABLE_FEATURE_SIZE, TABLE_ANGULAR_OFFSET, scaledBoundary, tableBoundary, tableRotation, tableYaw, wrapYaw } from './motion_table_features';
import type { SupportQuality } from './motion_quality';
import { Vector3 } from 'three';
import type { TableDocument, TableEntry, TableMotion } from './motion_table_types';

export type TableAsset = {
  entry: TableEntry; clip: SmplClip; scale: number; rate: number; begin: number; end: number;
  first: number[]; last: number[]; track: number[][]; length: number; yaw: number; avatarQuality?: SupportQuality; verticalOffset?: number;
};
function validate(table: TableDocument): void {
  if (!table || table.version !== 1 || table.fps !== 20 || table.up !== 'yup'
    || JSON.stringify(table.joints) !== JSON.stringify(TABLE_JOINTS) || table.bones?.length !== 27
    || table.parents?.length !== 27 || table.restJoints?.length !== 81 || !table.restJoints.every(Number.isFinite)
    || !/^[a-z0-9-]+\.bin$/.test(table.dataFile) || !/^[a-f0-9]{64}$/.test(table.dataSha256)
    || !Number.isInteger(table.dataFloats) || table.dataFloats <= 0 || !table.entries?.length
    || table.entries.length > 10000 || table.successors?.length !== table.entries.length) throw new Error('动作表格式无效');
  const ids = new Set<string>(); let offset = 0;
  for (const [index, entry] of table.entries.entries()) {
    if (!entry || !['idle', 'start', 'travel', 'stop', 'turn', 'action'].includes(entry.role)
      || ids.has(entry.id) || !entry.id || !Number.isInteger(entry.frames) || entry.frames < 12 || entry.frames > 2400
      || entry.fps !== 20 || entry.offset !== offset || !Number.isFinite(entry.sourceHeight) || entry.sourceHeight <= 0
      || !Number.isFinite(entry.heightOffset) || entry.entry?.length !== TABLE_FEATURE_SIZE || entry.exit?.length !== TABLE_FEATURE_SIZE
      || ![...entry.entry, ...entry.exit, entry.length, entry.yaw, entry.duration].every(Number.isFinite)
      || entry.trajectory?.length !== 9 || entry.trajectory.some(row => row.length !== 4 || !row.every(Number.isFinite))
      || table.successors[index].some(edge => !Number.isInteger(edge.to) || edge.to < 0 || edge.to >= table.entries.length
        || !Number.isFinite(edge.cost))) throw new Error(`动作表条目无效：${index}`);
    for (const feature of [entry.entry, entry.exit]) for (let j = 5; j < TABLE_ANGULAR_OFFSET; j += 4) {
      if (Math.abs(Math.hypot(...feature.slice(j, j+4))-1) > .001) throw new Error(`动作表关节旋转无效：${entry.id}`);
    }
    ids.add(entry.id); offset += entry.frames*88;
  }
  if (offset !== table.dataFloats) throw new Error('动作表数据边界不符');
}
function decode(table: TableDocument, values: Float32Array, entry: TableEntry): TableMotion {
  const start = entry.offset, end = start+entry.frames*88;
  const slice = (a: number, b: number): number[] => Array.from(values.subarray(a, b));
  return { version: 1, name: entry.name, source: entry.source, dataset: '检索动作表', up: 'yup', fps: 20,
    frames: entry.frames, duration: entry.duration, bones: table.bones, parents: table.parents, restJoints: table.restJoints,
    localRot: slice(start, start+entry.frames*81), rootDelta: slice(start+entry.frames*81, start+entry.frames*84),
    footContacts: slice(start+entry.frames*84, end), rootOrigin: [0, .92, 0],
    sourceHeight: entry.sourceHeight, heightOffset: entry.heightOffset };
}
function asset(entry: TableEntry, clip: SmplClip, scale: number, begin = 0): TableAsset {
  const end = clip.frames-1, origin = clip.rootAt(begin, new Vector3()), yaw = tableYaw(tableRotation(clip, begin, 0));
  const rate = Math.min(1, .5/Math.max(.01, entry.length*scale/entry.duration),
    entry.role === 'action' ? .35/Math.max(.01, entry.quality.angularStepMax) : 1);
  let length = 0, previous = origin;
  const track = Array.from({ length: 9 }, (_, i) => {
    const frame = begin+(end-begin)*i/8, point = clip.rootAt(frame, new Vector3());
    length += Math.hypot(point.x-previous.x, point.z-previous.z)*scale; previous = point;
    const local = point.clone().sub(origin).applyAxisAngle(new Vector3(0, 1, 0), -yaw).multiplyScalar(scale);
    return [local.x, local.z, wrapYaw(tableYaw(tableRotation(clip, frame, 0))-yaw), length];
  });
  return { entry, clip, scale, rate, begin, end, first: scaledBoundary(tableBoundary(clip, begin), scale, rate),
    last: scaledBoundary(entry.exit, scale, rate), track, length, yaw: track[8][2] };
}

export class MotionTableData {
  private constructor(readonly document: TableDocument, readonly assets: TableAsset[]) {}
  static async load(calibrate: (clip: SmplClip) => number | null, fallback: (height: number) => number): Promise<MotionTableData> {
    const response = await fetch('/motion-table/table.json');
    if (!response.ok) throw new Error(`动作表读取失败 HTTP ${response.status}`);
    const table = await response.json() as TableDocument; validate(table);
    const binary = await fetch(`/motion-table/${table.dataFile}`);
    if (!binary.ok) throw new Error(`动作表姿态读取失败 HTTP ${binary.status}`);
    const buffer = await binary.arrayBuffer();
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)), v => v.toString(16).padStart(2, '0')).join('');
    if (buffer.byteLength !== table.dataFloats*4 || digest !== table.dataSha256) throw new Error('动作表校验和不符');
    const values = new Float32Array(buffer);
    if (!values.every(Number.isFinite)) throw new Error('动作表包含无效姿态');
    const assets: TableAsset[] = [];
    for (const entry of table.entries) {
      const clip = SmplClip.from(decode(table, values, entry));
      const scale = calibrate(clip) ?? fallback(entry.sourceHeight);
      assets.push(asset(entry, clip, scale));
      const starts = entry.role === 'travel' ? [4, 8, 12] : entry.role === 'stop' ? [8, 16, 24] : [];
      for (const begin of starts) {
        if (begin < clip.frames-10) assets.push(asset(entry, clip, scale, begin));
      }
    }
    return new MotionTableData(table, assets);
  }
}

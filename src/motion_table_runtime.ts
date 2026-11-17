import type { VRM } from '@pixiv/three-vrm';
import type { MotionSeed, PresetResult } from './locomotion';
import { measureSupport, type SupportQuality } from './motion_quality';
import type { MotionTableData } from './motion_table_data';
import { searchMotionTable, TableCoverageError, type TableRoute } from './motion_table_search';
import { measureTablePath } from './motion_table_quality';
import { assembleMotionTable, TableCollisionError, type TableSafeRoot } from './motion_table_playback';

type Candidate = { result: PresetResult; quality: SupportQuality };
function evaluate(vrm: VRM, id: string, route: TableRoute, seed: MotionSeed, safe: TableSafeRoot, path: number[][]): Candidate {
  const result = assembleMotionTable(id, route, seed, safe);
  const tracking = measureTablePath(result.motion, path, seed.motion.frames);
  result.table = { ...route.diagnostics, pathMaxError: tracking.maximum, goalError: tracking.goal };
  const quality = measureSupport(vrm, result.motion, result.phases, seed.motion.frames, true);
  return { result, quality };
}
const acceptable = (q: SupportQuality): boolean => q.samples >= 4 && q.mean <= .12 && q.p95 <= .32;
const preferred = (q: SupportQuality): boolean => acceptable(q) && q.mean <= .09 && q.p95 <= .25;
const score = (q: SupportQuality): number => q.mean+.25*q.p95;

export function compileMotionTable(vrm: VRM, table: MotionTableData, id: string, path: number[][],
  seed: MotionSeed, safe: TableSafeRoot): Candidate {
  const route = searchMotionTable(table.assets, path, seed);
  let best: Candidate | null = null, detail = '', collision: TableCollisionError | null = null;
  for (const alternative of [route, ...route.alternatives ?? []]) {
    let current: Candidate;
    try { current = evaluate(vrm, id, alternative, seed, safe, path); }
    catch (error) {
      if (error instanceof TableCollisionError) { collision = error; continue; }
      if (!(error instanceof TableCoverageError)) throw error;
      detail = error.message; continue;
    }
    const { quality } = current;
    if (acceptable(quality) && (!best || score(quality) < score(best.quality))) best = current;
    if (preferred(quality)) return current;
    detail = `脚部校验均值 ${quality.mean.toFixed(3)} / P95 ${quality.p95.toFixed(3)} m/s`;
  }
  if (best) return best;
  if (collision) throw collision;
  throw new TableCoverageError(`动作表未通过当前角色质量门限：${detail}`);
}

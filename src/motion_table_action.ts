import { Vector3 } from 'three';
import type { VRM } from '@pixiv/three-vrm';
import { SmplClip } from './retarget';
import type { MotionSeed, PresetResult } from './locomotion';
import type { MotionTableData } from './motion_table_data';
import { tableBoundary, tableCost, tableRotation, tableYaw } from './motion_table_features';
import { TableCoverageError } from './motion_table_search';
import { assembleMotionTable, TableTransitionError, type TableSafeRoot } from './motion_table_playback';
import { measureSupport } from './motion_quality';

/** Resolve a complete semantic gesture with the same full-body seam mechanism. */
export function compileTableAction(vrm: VRM, table: MotionTableData, prompt: string, seed: MotionSeed,
  safe: TableSafeRoot, uses: Map<string, number> = new Map()): PresetResult {
  const clip = SmplClip.from(seed.motion), feature = tableBoundary(clip);
  const position = clip.rootAt(clip.frames-1, new Vector3()), heading = tableYaw(tableRotation(clip, clip.frames-1, 0));
  const candidates = table.assets.filter(asset => asset.entry.role === 'action' && asset.entry.prompt === prompt
    && asset.entry.duration >= 2 && asset.begin === 0).map(asset => ({ asset, seam: tableCost(feature, asset.first) }))
    .filter(row => row.seam.rms <= .45 && row.seam.maximum <= 1.2 && row.seam.velocity <= .18)
    .sort((a, b) => a.seam.cost+.015*(uses.get(a.asset.entry.id) ?? 0)-b.seam.cost-.015*(uses.get(b.asset.entry.id) ?? 0));
  for (const { asset, seam } of candidates) {
    const route = { parts: [{ asset, x: position.x, z: position.z, heading }], diagnostics: {
      entries: [asset.entry.id], searchMs: 0, expanded: candidates.length, pathMaxError: 0, goalError: 0,
      seamPoseRms: seam.rms, seamJointMax: seam.maximum, seamVelocityMax: seam.velocity } };
    let result: PresetResult;
    try { result = assembleMotionTable(asset.entry.id, route, seed, safe); }
    catch (error) {
      if (!(error instanceof TableTransitionError)) throw error;
      continue;
    }
    const quality = measureSupport(vrm, result.motion, result.phases, seed.motion.frames, true);
    if (quality.samples >= 4 && quality.mean <= .12 && quality.p95 <= .32) {
      uses.set(asset.entry.id, (uses.get(asset.entry.id) ?? 0)+1);
      return result;
    }
  }
  throw new TableCoverageError('此手势缺少可从当前姿态平滑接入的完整动作；需要补充采集');
}

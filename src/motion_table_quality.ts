import type { SmplMotionJson } from './retarget';
import { TableCoverageError } from './motion_table_search';

/** Validate every emitted root sample, including the inertialized handoff. */
export function measureTablePath(motion: SmplMotionJson, path: number[][], begin: number): { maximum: number; goal: number } {
  const roots = motion.rootDelta, start = roots.slice(Math.max(0, begin-1)*3, Math.max(0, begin-1)*3+3);
  const points = [start, ...path].filter((point, index, list) => index === 0
    || Math.hypot(point[0]-list[index-1][0], point[2]-list[index-1][2]) > .001);
  if (points.length < 2) throw new TableCoverageError('动作表路线缺少有效移动');
  let maximum = 0, progress = 0;
  const lengths = points.slice(1).map((point, index) => Math.hypot(point[0]-points[index][0], point[2]-points[index][2]));
  const arc = [0]; for (const length of lengths) arc.push(arc.at(-1)!+length);
  for (let frame = begin; frame < motion.frames; frame++) {
    const x = roots[frame*3], z = roots[frame*3+2];
    let closest = Infinity, chosen = progress;
    for (let i = 1; i < points.length; i++) {
      if (arc[i] < progress-.08 || arc[i-1] > progress+3) continue;
      const a = points[i-1], b = points[i], dx = b[0]-a[0], dz = b[2]-a[2];
      const t = Math.max(0, Math.min(1, ((x-a[0])*dx+(z-a[2])*dz)/(dx*dx+dz*dz)));
      const distance = Math.hypot(x-a[0]-dx*t, z-a[2]-dz*t);
      if (distance < closest) { closest = distance; chosen = arc[i-1]+lengths[i-1]*t; }
    }
    progress = Math.max(progress, chosen); maximum = Math.max(maximum, closest);
  }
  const target = points.at(-1)!;
  const goal = Math.hypot(roots.at(-3)!-target[0], roots.at(-1)!-target[2]);
  if (maximum > .3 || goal > .12) throw new TableCoverageError(`实际根轨迹超出门限：路径 ${maximum.toFixed(3)}m / 终点 ${goal.toFixed(3)}m`);
  return { maximum, goal };
}

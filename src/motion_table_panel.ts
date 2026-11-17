import type { TableEntry } from './motion_table_types';
import type { MotionTableData } from './motion_table_data';

const LABELS = { idle: '待机', start: '起步', travel: '移动', stop: '收步', turn: '转身', action: '交互动作' };
/** Research catalog: source provenance and quality are visible without inference. */
export function showMotionTable(table: MotionTableData): void {
  const select = document.querySelector<HTMLSelectElement>('#motion-table-select')!;
  const metadata = document.querySelector('#motion-table-entry')!;
  const rows = table.document.entries;
  select.replaceChildren(...rows.map(entry => {
    const option = document.createElement('option'); option.value = entry.id;
    option.textContent = `${LABELS[entry.role]} · ${entry.prompt || entry.name} · ${entry.duration.toFixed(2)}s`;
    return option;
  }));
  const render = (entry: TableEntry): void => {
    metadata.textContent = `${entry.source} · ${entry.sourceFrames?.join('–') ?? ''} 帧\n`
      + `原始根位移 ${entry.length.toFixed(2)}m · 转角 ${(entry.yaw*180/Math.PI).toFixed(1)}° · 源骨架滑动 P95 ${entry.quality.supportP95.toFixed(3)}m/s`;
  };
  select.onchange = () => { const entry = rows.find(row => row.id === select.value); if (entry) render(entry); };
  render(rows[0]);
  document.querySelector('#motion-table-status')!.textContent = `${rows.length} 段筛选动作 · ${rows.filter(entry => entry.role === 'travel').length} 段移动 · ${table.document.successors.reduce((sum, edges) => sum+edges.length, 0)} 条候选接缝`;
}

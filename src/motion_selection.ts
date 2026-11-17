type SelectionState = { counts: Record<string, number>; recent: string[] };
const STORAGE_KEY = 'motion-variant-use-v1';

/** Pose compatibility first, then avoid repetition among acceptable candidates. */
export class VariantMemory {
  private state: SelectionState = { counts: {}, recent: [] };

  constructor() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (!saved) return;
      const data = JSON.parse(saved) as SelectionState;
      if (!Array.isArray(data.recent) || !data.counts || Object.values(data.counts).some(n => !Number.isFinite(n) || n < 0)) {
        throw new Error('动作使用记录格式无效');
      }
      this.state = data;
    } catch (error) { console.warn('动作变体记录不可用，将使用本次会话记录', error); }
  }

  choose<T>(candidates: T[], name: (item: T) => string, cost: (item: T) => number): T {
    if (!candidates.length) throw new Error('没有可用的动作变体');
    const scored = candidates.map(item => ({ item, name: name(item), cost: cost(item) }));
    const best = Math.min(...scored.map(entry => entry.cost));
    const compatible = scored.filter(entry => entry.cost <= best+Math.max(.18, best*.35));
    compatible.sort((a, b) => {
      const recent = (id: string): number => this.state.recent.includes(id) ? this.state.recent.lastIndexOf(id)+1 : 0;
      return recent(a.name)-recent(b.name) || (this.state.counts[a.name] ?? 0)-(this.state.counts[b.name] ?? 0) || a.cost-b.cost;
    });
    return compatible[0].item;
  }

  commit(names: string[]): void {
    for (const name of names) {
      this.state.counts[name] = (this.state.counts[name] ?? 0)+1;
      this.state.recent = [...this.state.recent.filter(value => value !== name), name].slice(-8);
    }
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(this.state)); }
    catch (error) { console.warn('无法持久保存动作变体记录，本次会话继续使用', error); }
  }
}

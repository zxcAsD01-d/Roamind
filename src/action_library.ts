import type { SmplMotionJson } from './retarget';

export type LibraryClip = { id: string; action: string; duration: number; plays: number;
  quality: { supportMean: number | null; supportP95: number | null; angularStepMax: number } };
export type LibraryCatalog = { clips: LibraryClip[]; usage: { action: string; uses: number; error: string }[];
  maxVariants: number; variantUseThresholds: number[] };
export type LibrarySelection = { action: string; duration: number; clipId?: string };
export type LibraryMatch = { hit: boolean; reason?: string; detail?: string; clipId?: string;
  motion?: SmplMotionJson; variants: number; targetVariants: number };
type MotionCapabilities = { actions: { action: string; label: string; channel: string; variants?: number }[] };
const ACTIONS: Record<string, string> = { wave: '挥手', raise: '举手', point: '指向', crouch: '蹲下', squat: '下蹲', sit: '坐下' };

export const actionLibraryPanel = `<details id="action-library-panel" class="trajectory-editor" open>
  <summary>动作变体 / 播放库</summary>
  <label class="field-label" for="library-select">自动选片或指定已存变体</label>
  <select id="library-select"></select>
  <label class="field-label" for="library-duration">动作时长（秒）</label>
  <input id="library-duration" type="number" value="3" min="1" max="30" step="0.05" />
  <button id="play-library-action" class="secondary-button">执行动作 / 缺失则生成</button>
  <p id="library-status" class="trajectory-note">读取动作库…</p>
</details>
<details id="motion-table-panel" class="trajectory-editor" open>
  <summary>连续动作表</summary>
  <p id="motion-table-status" class="trajectory-note">读取全身动作索引…</p>
  <p class="trajectory-note">曲线路线自动检索整段步态，结合关节姿态、角速度与支撑脚选择接缝。可试走前方曲线；规划中的曲线路线会自动接入。</p>
  <label class="field-label" for="motion-table-select">动作表来源与质量</label>
  <select id="motion-table-select"></select>
  <p id="motion-table-entry" class="trajectory-note"></p>
  <button id="motion-table-gesture" class="secondary-button">执行所选类别的完整手势</button>
  <button id="motion-table-test" class="secondary-button">试走前方曲线</button>
  <p id="motion-table-result" class="trajectory-note" aria-live="polite"></p>
</details>`;

export class ActionLibraryPanel {
  private clips: LibraryClip[] = [];
  constructor(private base: string, private play: (selection: LibrarySelection) => Promise<void>,
    private refreshed: (clips: LibraryClip[]) => void) {
    document.querySelector('#library-select')!.addEventListener('change', () => {
      const selected = this.selected();
      if (selected.clipId) document.querySelector<HTMLInputElement>('#library-duration')!.value = String(selected.duration);
    });
    document.querySelector('#play-library-action')!.addEventListener('click', () => { void this.execute(); });
  }

  private selected(): LibrarySelection {
    const id = document.querySelector<HTMLSelectElement>('#library-select')!.value;
    const clip = this.clips.find(value => value.id === id);
    const duration = clip?.duration ?? Number(document.querySelector<HTMLInputElement>('#library-duration')!.value);
    if (!Number.isFinite(duration) || duration < 1 || duration > 30) throw new Error('动作时长须在 1–30 秒内');
    return { action: clip?.action ?? id, duration, ...(clip ? { clipId: clip.id } : {}) };
  }

  private async execute(): Promise<void> {
    const status = document.querySelector('#library-status')!;
    try { await this.play(this.selected()); await this.refresh(); }
    catch (error) { status.textContent = error instanceof Error ? error.message : '动作执行失败'; }
  }

  async refresh(): Promise<void> {
    const [response, capabilitiesResponse] = await Promise.all([fetch(`${this.base}/api/library`),
      fetch(`${this.base}/api/actions/capabilities`)]);
    if (!response.ok) throw new Error(`动作库读取失败：HTTP ${response.status}`);
    if (!capabilitiesResponse.ok) throw new Error(`动作能力目录读取失败：HTTP ${capabilitiesResponse.status}`);
    const catalog = await response.json() as LibraryCatalog;
    const capabilities = await capabilitiesResponse.json() as MotionCapabilities;
    this.clips = catalog.clips;
    const select = document.querySelector<HTMLSelectElement>('#library-select')!, previous = select.value;
    const option = (value: string, label: string): HTMLOptionElement => {
      const element = document.createElement('option'); element.value = value; element.textContent = label; return element;
    };
    const automatic = Object.entries(ACTIONS).map(([action, label]) => option(action, `${label} · 自动选片`));
    const saved = this.clips.filter(clip => clip.action in ACTIONS).map(clip => option(clip.id,
      `${ACTIONS[clip.action]} · ${clip.id.slice(-6)} · ${clip.duration}s · 已播 ${clip.plays} 次`));
    const native = capabilities.actions.map(row => option(row.action, `${row.channel === 'full-body' ? 'AMASS 全身' : '流明'} · ${row.label}${row.variants ? ` · ${row.variants} 变体` : ''}`));
    select.replaceChildren(...automatic, ...native, ...saved);
    if ([...select.options].some(item => item.value === previous)) select.value = previous;
    const rejected = catalog.usage.filter(entry => entry.error).length;
    document.querySelector('#library-status')!.textContent = `${native.length} 类可执行动作 · 吃喝各 3 套节奏 · ${this.clips.length} 条生成库变体`
      + (rejected ? ` · ${rejected} 类动作有入库拒绝记录，详见 MCP 动作库状态` : '');
    this.refreshed(this.clips);
  }
}

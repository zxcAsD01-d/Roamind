export type InteractionOption = { id: string; operation: string; label: string; enabled: boolean;
  reason: string; score: number; stages: string[] };
export type InteractionObject = { objectId: string; label: string; position: number[]; frameId: number;
  observedAt: number; confidence: number; distance: number | null; fresh: boolean; options: InteractionOption[] };
export type InteractionCatalog = { objects: InteractionObject[]; planner: string; source: string };
type Context = { clientId: string; request: <T>(path: string, body: object) => Promise<T>;
  submitted: () => void };
export const objectInteractionPanel =
  '<details class="runtime-interaction-options" open><summary>视觉物体 · 自动交互选项</summary>'+
  '<p class="ardy-status">双目表面与语义记忆 → 接近、坐卧、拿取、吃喝或玩电脑。食物和水杯可沿记忆寻找，接触前须重新识别。</p>'+
  '<label class="field-label" for="interaction-object">识别目标</label><select id="interaction-object"></select>'+
  '<div id="interaction-options" class="scene-buttons"></div>'+
  '<p id="interaction-status" class="ardy-status" role="status">等待同帧语义和深度；坐卧需要支撑面，拿取需要实测抓取点。</p></details>';

export class ObjectInteractions {
  private catalog: InteractionCatalog = { objects: [], planner: '', source: '' };
  private pending = false;
  private signature = '';
  private noticeUntil = 0;
  constructor(private context: Context) {
    document.querySelector('#interaction-object')!.addEventListener('change', () => { this.signature = ''; this.render(); });
  }
  sync(catalog: InteractionCatalog): void { this.catalog = catalog; this.render(); }

  private render(): void {
    const select = document.querySelector<HTMLSelectElement>('#interaction-object')!;
    const selected = select.value;
    const signature = JSON.stringify(this.catalog.objects.map(row => [row.objectId, row.fresh,
      row.distance === null ? null : Math.round(row.distance*10), row.options.map(option => [option.enabled, option.reason])]))+selected;
    if (signature === this.signature) return;
    this.signature = signature;
    const rows = this.catalog.objects;
    select.replaceChildren(...rows.map(row => {
      const option = document.createElement('option'); option.value = row.objectId;
      option.textContent = row.label+' · '+(row.distance === null ? '距离未知' : row.distance.toFixed(1)+'m')+(row.fresh ? '' : ' · 历史');
      return option;
    }));
    if (rows.some(row => row.objectId === selected)) select.value = selected;
    const row = rows.find(row => row.objectId === select.value);
    document.querySelector('#interaction-options')!.replaceChildren(...(row?.options ?? []).map(option => {
      const button = document.createElement('button'); button.className = 'secondary-button';
      button.textContent = option.label; button.disabled = !option.enabled || this.pending;
      button.title = option.reason || option.stages.join(' → ');
      button.addEventListener('click', () => { void this.submit(row!.objectId, option.operation); });
      return button;
    }));
    if (!this.pending && performance.now() >= this.noticeUntil) this.status(rows.length ? rows.length+' 个视觉物体 · 选择目标后执行可用交互' : '等待同帧语义和深度');
  }

  private async submit(objectId: string, operation: string): Promise<void> {
    if (this.pending) return;
    this.pending = true; this.signature = ''; this.render();
    try {
      await this.context.request('/api/interactions/submit', { clientId: this.context.clientId, objectId, operation });
      this.context.submitted();
      this.status('交互任务已提交，按观察→接近→执行持续推进');
    } catch (error) {
      this.status(error instanceof Error ? error.message : '交互任务提交失败');
    } finally {
      this.noticeUntil = performance.now()+5000;
      this.pending = false; this.signature = ''; this.render();
    }
  }
  private status(message: string): void { document.querySelector('#interaction-status')!.textContent = message; }
}

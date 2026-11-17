import * as THREE from 'three';
import type { EgoPerception } from './ego_stereo';
import { perceptionRequest } from './perception';
import type { TouchArgs, TouchController } from './touch';
import type { TouchTarget } from './touch_contact';
import type { VRM } from '@pixiv/three-vrm';
import { selfEffector, type EffectorName } from './contact_rig';

export type OcrArgs = { text: string; heading?: number; direction?: string; occurrence?: number };
type TextBox = { text: string; confidence: number; box: number[]; center: number[] };
type OcrResult = { status: string; candidates: TextBox[]; selected: TextBox | null; target?: TouchTarget;
  observedAt: number; frameId: number; inferenceMs: number };
type Context = { ego: () => EgoPerception | null; touch: () => TouchController | null;
  prepare: (heading: number | undefined, current: () => boolean) => Promise<void>;
  request: <T>(path: string, body?: object) => Promise<T>; scene: THREE.Scene;
  avatar: THREE.Group; heading: () => number; vrm: () => VRM | null };

export const ocrPanel = `<section id="visual-text-panel" class="perception-panel"><h3>文字标注选择</h3>
  <label class="field-label" for="ocr-text">目标文字</label><input id="ocr-text" class="scene-input" value="确认">
  <label class="field-label" for="ocr-heading">朝向（度，留空保持当前方向）</label><input id="ocr-heading" class="scene-input" type="number" step="15" placeholder="0 朝 +Z，90 朝 +X">
  <label class="field-label" for="ocr-direction">文字位于视野</label><select id="ocr-direction"><option value="anywhere">任意位置</option><option value="left">左侧</option><option value="right">右侧</option><option value="up">上方</option><option value="down">下方</option></select>
  <label class="field-label" for="ocr-occurrence">重复文字序号（从 0 起，留空要求唯一）</label><input id="ocr-occurrence" class="scene-input" type="number" min="0" step="1" placeholder="唯一匹配">
  <div class="scene-buttons"><button id="ocr-scan" class="secondary-button">识别当前视野</button><button id="ocr-touch" class="secondary-button">转向并选择文字标注</button></div>
  <button id="ocr-board" class="secondary-button">添加 / 移除文字测试板</button>
  <p id="ocr-status" class="ardy-status" aria-live="polite">OCR 仅辅助选择目标；选中后使用统一触碰操作。</p>
  <ol id="ocr-results" class="plan-timeline"></ol>
</section>`;

export class TextContactController {
  private board: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial> | null = null;
  private busy = false;

  constructor(private context: Context) {
    this.bind('ocr-scan', async () => { await this.scan(this.args()); });
    this.bind('ocr-touch', () => this.selectViewTarget());
    this.bind('ocr-board', async () => this.toggleBoard());
  }

  get active(): boolean { return this.busy; }

  finish(status: string, detail: string): void {
    this.status(status === 'completed' ? '文字触碰完成 · 已撤回；按钮业务状态需另行确认'
      : `文字触碰${status === 'failed' ? '失败' : '已打断'}${detail ? ' · '+detail : ''}`);
  }

  private bind(id: string, task: () => Promise<void>): void {
    document.querySelector(`#${id}`)!.addEventListener('click', () => {
      void task().catch(error => this.status(error instanceof Error ? error.message : '文字操作失败'));
    });
  }

  private args(): OcrArgs {
    const value = (id: string): string => document.querySelector<HTMLInputElement | HTMLSelectElement>(`#${id}`)!.value.trim();
    return { text: value('ocr-text'), direction: value('ocr-direction'),
      ...(value('ocr-heading') ? { heading: Number(value('ocr-heading')) } : {}),
      ...(value('ocr-occurrence') ? { occurrence: Number(value('ocr-occurrence')) } : {}) };
  }

  private async selectViewTarget(): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      const args = this.args();
      this.context.ego()?.setEnabled(true);
      await this.prepareSelection(args.heading);
      const after = Date.now()/1000;
      await this.waitForFrame(after, () => true);
      const result = await this.scan(args, true, after);
      if (!result.target) throw new Error('目标未找到或不唯一，请选择识别结果');
      const touch = this.context.touch();
      if (!touch) throw new Error('触碰控制器尚未就绪');
      touch.selectTarget(result.target);
      this.status('已选中文字标注；使用统一触碰操作执行');
    } finally { this.busy = false; }
  }

  private async prepareSelection(heading?: number): Promise<void> {
    const path = heading === undefined ? '/api/sim/interrupt' : '/api/sim/face';
    const command = await this.context.request<{ id: string }>(path, heading === undefined ? { mode: 'hold' } : { heading });
    const deadline = performance.now()+30000;
    while (performance.now() < deadline) {
      const state = await this.context.request<{ status: string; detail?: string }>(`/api/sim/commands/${command.id}`);
      if (state.status === 'completed') return;
      if (['failed', 'interrupted', 'superseded'].includes(state.status)) {
        throw new Error(state.detail || '标注选择前的转向已打断');
      }
      await new Promise<void>(resolve => setTimeout(resolve, 100));
    }
    throw new Error('标注选择前的转向或停步超时');
  }

  private async selectBox(item: TextBox, frameId: number): Promise<void> {
    const ego = this.context.ego(), touch = this.context.touch();
    if (!ego || !touch) throw new Error('视觉或触碰控制器尚未就绪');
    const target = await perceptionRequest<TouchTarget>('/api/touch/resolve', {
      clientId: ego.sessionId, pixel: item.center, frameId, label: item.text });
    touch.selectTarget(target);
    this.status(`已选择“${item.text}”标注；使用统一触碰操作执行`);
  }

  async execute(args: OcrArgs, current: () => boolean): Promise<void> {
    this.busy = true;
    try {
      this.context.ego()?.setEnabled(true);
      this.status('正在转向并停稳');
      await this.context.prepare(args.heading, current);
      if (!current()) return;
      const after = Date.now()/1000;
      await this.waitForFrame(after, current);
      if (!current()) return;
      const result = await this.scan(args, true, after);
      if (!current()) return;
      if (!result.target) throw new Error(`文字目标${result.status === 'ambiguous' ? '不唯一，请指定方向或序号' : '未找到'}，保持站立`);
      const touch = this.context.touch();
      if (!touch) throw new Error('触摸控制器尚未就绪');
      const touchArgs: TouchArgs = { selection: { clientId: this.context.ego()!.sessionId, targetId: result.target.targetId },
        effector: this.selectHand(result.target), operation: 'touch', duration: .6 };
      await touch.execute(touchArgs, current);
      this.status(`正在触碰“${args.text}” · 位置仅来自 OCR 与双目`);
    } finally { this.busy = false; }
  }

  private async waitForFrame(after: number, current: () => boolean): Promise<void> {
    const deadline = performance.now()+5000;
    while (current()) {
      const observation = this.context.ego()?.visualObservation();
      if (observation && observation.observedAt > after && Date.now()/1000-observation.observedAt < 1.5) return;
      if (performance.now() > deadline) throw new Error('转向后未获得新鲜双目帧，请检查第一人称感知');
      await new Promise<void>(resolve => setTimeout(resolve, 50));
    }
  }

  private selectHand(target: TouchTarget): EffectorName {
    const vrm = this.context.vrm();
    if (!vrm) throw new Error('VRM 尚未加载');
    const destination = new THREE.Vector3().fromArray(target.position).addScaledVector(new THREE.Vector3().fromArray(target.normal), .035);
    const hands = (['left_hand', 'right_hand'] as const).map(name => {
      const own = selfEffector(vrm, name), base = new THREE.Vector3().fromArray(own.base);
      const middle = new THREE.Vector3().fromArray(own.middle), end = new THREE.Vector3().fromArray(own.end);
      return { name, reach: destination.distanceTo(base)/(base.distanceTo(middle)+middle.distanceTo(end)) };
    });
    hands.sort((a, b) => a.reach-b.reach);
    if (hands[0].reach > .995) throw new Error('文字目标超出双手可达范围，请先移动后重新观测');
    return hands[0].name;
  }

  private async scan(args: OcrArgs, resolveDepth = false, afterObservedAt = 0): Promise<OcrResult> {
    const ego = this.context.ego();
    if (!ego) throw new Error('第一人称相机尚未就绪');
    ego.setEnabled(true);
    const result = await perceptionRequest<OcrResult>('/api/ocr/ego', {
      ...args, clientId: ego.sessionId, resolveDepth, afterObservedAt });
    const list = document.querySelector('#ocr-results')!;
    list.replaceChildren(...result.candidates.map((item, index) => {
      const row = document.createElement('li');
      row.textContent = `${index} · ${item.text} · ${(item.confidence*100).toFixed(0)}% · (${item.center.map(v => v.toFixed(0)).join(', ')}) px`;
      const select = document.createElement('button'); select.textContent = '选择标注'; select.className = 'secondary-button';
      select.addEventListener('click', () => {
        void this.selectBox(item, result.frameId).catch(error => this.status(error instanceof Error ? error.message : '选择文字标注失败'));
      });
      row.append(select);
      return row;
    }));
    document.querySelector<HTMLElement>('#ocr-status')!.dataset.result = JSON.stringify(result);
    this.status(`${result.candidates.length} 个匹配 · ${result.status} · ${result.inferenceMs.toFixed(0)} ms`);
    return result;
  }

  private toggleBoard(): void {
    if (this.board) {
      this.context.scene.remove(this.board); this.board.geometry.dispose();
      this.board.material.map?.dispose(); this.board.material.dispose(); this.board = null; return;
    }
    const canvas = document.createElement('canvas'); canvas.width = 768; canvas.height = 512;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#e8e5df'; ctx.fillRect(0, 0, 768, 512);
    // Aperiodic texture supports stereo; no label coordinates reach perception.
    let noise = 71347;
    for (let y = 0; y < 512; y += 3) for (let x = 0; x < 768; x += 3) {
      noise ^= noise << 13; noise ^= noise >>> 17; noise ^= noise << 5;
      const shade = 180+(noise >>> 0)%65;
      ctx.fillStyle = `rgb(${shade},${shade},${shade})`; ctx.fillRect(x, y, 3, 3);
    }
    ctx.fillStyle = '#1a2429'; ctx.font = 'bold 66px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('确认', 205, 280); ctx.fillText('取消', 565, 280);
    ctx.font = '32px sans-serif'; ctx.fillText('VISUAL TOUCH', 384, 85);
    const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
    this.board = new THREE.Mesh(new THREE.PlaneGeometry(.72, .48), new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide }));
    const heading = this.context.heading();
    this.board.position.copy(this.context.avatar.position).add(new THREE.Vector3(Math.sin(heading)*.41, 1.02, Math.cos(heading)*.41));
    this.board.rotation.y = heading+Math.PI;
    this.context.scene.add(this.board);
    this.status('已添加带真实纹理的文字板；OCR / 双目只处理相机图像');
  }

  private status(message: string): void { document.querySelector('#ocr-status')!.textContent = message; }
}

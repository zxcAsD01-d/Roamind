import type { VRM } from '@pixiv/three-vrm';
import type * as THREE from 'three';
import type { MotionSeed } from './locomotion';
import type { SmplMotionJson } from './retarget';
import { perceptionRequest, type EgoPerception } from './perception';
import { fitTouchContact, type TouchPlan, type TouchTarget } from './touch_contact';
import { selfEffector, type EffectorName } from './contact_rig';
import { LiveContact } from './contact_tracking';
import type { Collider } from './house';

export type TouchArgs = { selection: { clientId: string; targetId?: string; personId?: string; bodyPart?: string;
  pixel?: number[]; frameId?: number }; hand?: string; effector?: EffectorName; operation?: string; duration?: number; amplitude?: number };
type TargetFrame = { image: string; frameId: number; observedAt: number };
type TouchJob = { id: string; status: 'generating' | 'ready' | 'failed' | 'cancelled'; error?: string;
  motion: SmplMotionJson | null; plan: TouchPlan };
type Context = { ego: () => EgoPerception | null; request: <T>(path: string, body?: object) => Promise<T>;
  stabilize: (current: () => boolean) => Promise<void>; seed: () => MotionSeed; vrm: () => VRM | null;
  root: THREE.Group; play: (motion: SmplMotionJson, seed: MotionSeed) => void;
  stop: (reason: string) => void; addPartner: () => Promise<void>; obstacles: () => Collider[] };
const parts = [['nose', '鼻部'], ['left_ear', '左耳关键点'], ['right_ear', '右耳关键点'],
  ['left_shoulder', '左肩'], ['right_shoulder', '右肩'], ['left_elbow', '左肘'], ['right_elbow', '右肘'],
  ['left_wrist', '左腕'], ['right_wrist', '右腕'], ['left_hip', '左髋'], ['right_hip', '右髋'],
  ['left_knee', '左膝'], ['right_knee', '右膝'], ['left_ankle', '左踝'], ['right_ankle', '右踝'],
  ['left_forearm', '左前臂表面（推断）'], ['right_forearm', '右前臂表面（推断）'],
  ['left_thigh', '左大腿表面（推断）'], ['right_thigh', '右大腿表面（推断）']] as const;

export const touchPanel = `<section id="visual-contact-panel" class="perception-panel"><p class="eyebrow">VISUAL CONTACT</p><h2>通用视觉接触</h2>
  <button id="touch-partner" class="secondary-button">添加 / 移除触摸测试角色</button>
  <label class="check-row"><input id="touch-partner-moving" type="checkbox">测试目标缓慢平移（仅用于视觉跟踪验证）</label>
  <label class="field-label" for="touch-partner-range">测试平移幅度（米）</label><input id="touch-partner-range" class="scene-input" type="number" min=".05" max="1.5" step=".01" value=".08">
  <label class="field-label" for="touch-label">标注名称</label><input id="touch-label" class="scene-input" value="视觉标注" maxlength="120">
  <canvas id="touch-image" width="640" height="360" aria-label="点击双目画面任意可见位置标注目标" style="width:100%;cursor:crosshair"></canvas>
  <p class="ardy-status">点击画面任意可见位置选择目标；人体和文字检测也可辅助选择同一类接触目标。</p>
  <details><summary>通过人体部位选择目标</summary>
  <label class="field-label" for="touch-person">检测到的人物</label><select id="touch-person"><option value="">等待双目人体检测</option></select>
  <label class="field-label" for="touch-part">身体部位</label><select id="touch-part">${parts.map(([key, name]) => `<option value="${key}">${name}</option>`).join('')}</select>
  <button id="touch-pick-part" class="secondary-button">选中部位标注</button></details>
  <details id="touch-ocr-selection" open><summary>通过文字选择目标</summary></details>
  <label class="field-label" for="touch-target">当前接触目标</label><select id="touch-target"><option value="">请先选择标注位置</option></select>
  <div class="scene-buttons"><select id="touch-hand" aria-label="接触执行部位"><option value="right_hand">右手</option><option value="left_hand">左手</option><option value="head">头部</option><option value="right_foot">右脚（需支撑姿态）</option><option value="left_foot">左脚（需支撑姿态）</option></select>
  <select id="touch-operation" aria-label="触摸操作"><option value="touch">轻触</option><option value="tap">轻拍一次</option><option value="hold">停留</option><option value="stroke">局部滑动</option></select></div>
  <label class="field-label" for="touch-duration">停留时间（秒）</label><input id="touch-duration" class="scene-input" type="number" min=".2" max="3" step=".1" value=".6">
  <label class="field-label" for="touch-amplitude">滑动幅度（米）</label><input id="touch-amplitude" class="scene-input" type="number" min="0" max=".04" step=".005" value=".01">
  <div class="scene-buttons"><button id="touch-start" class="primary-button">触碰所选目标</button><button id="touch-stop" class="secondary-button">停止触摸</button></div>
  <p id="touch-status" class="ardy-status" aria-live="polite">先开启双目视觉。接触使用骨骼球形锚点，无手指抓握或承重仿真。</p>
  <p id="contact-live" class="ardy-status" aria-live="polite">播放时跟踪视觉位置；只在选定骨骼链内修正。</p>
  </section>`;

export class TouchController {
  private job: string | null = null;
  private revision = 0;
  private selected: TouchTarget | null = null;
  private targets = new Map<string, TouchTarget>();
  private rendered: TargetFrame | null = null;
  private imageRevision = 0;
  private watching: TouchPlan | null = null;
  private watchingBusy = false;
  private live: LiveContact | null = null;
  private contactSamples = 0;
  private reachedSamples = 0;

  constructor(private context: Context) {
    this.bind('touch-start', () => this.enqueue());
    this.bind('touch-stop', async () => { this.cancel(); await this.context.request('/api/sim/interrupt', { mode: 'hold' }); });
    this.bind('touch-partner', () => this.context.addPartner());
    this.bind('touch-pick-part', () => this.selectPart());
    document.querySelector<HTMLSelectElement>('#touch-target')!.addEventListener('change', event => {
      this.selected = this.targets.get((event.target as HTMLSelectElement).value) ?? null;
      this.renderSelection();
    });
    document.querySelector<HTMLCanvasElement>('#touch-image')!.addEventListener('click', event => {
      void this.selectPixel(event).catch(error => this.status(error instanceof Error ? error.message : '选点失败'));
    });
    window.setInterval(() => { this.refresh(); void this.watch(); }, 250);
  }

  private bind(id: string, callback: () => Promise<void>): void {
    document.querySelector(`#${id}`)!.addEventListener('click', () => {
      void callback().catch(error => this.status(error instanceof Error ? error.message : '触摸操作失败'));
    });
  }

  get active(): boolean { return !!(this.job || this.watching || this.live); }

  completionError(): string | null {
    return this.reachedSamples >= 2 ? null : '接触播放结束，但未获得至少两次有效几何接触采样';
  }

  finish(status: string, detail: string): void {
    if (status === 'completed') {
      this.status(`视觉接触完成 · ${this.reachedSamples} 次有效采样 · 已撤回`);
      document.querySelector('#contact-live')!.textContent = `已撤回 · 接触阶段有效采样 ${this.reachedSamples}/${this.contactSamples}；当前手部间距不作为接触误差`;
    } else this.status(detail || '视觉接触已打断');
  }

  cancel(): void {
    const id = this.job ?? this.watching?.id;
    this.revision++; this.watching = null; this.live = null; this.job = null;
    if (id) void this.context.request('/api/touch/cancel', { id }).catch(error => {
      this.status(error instanceof Error ? error.message : '取消触摸失败');
    });
  }

  private async enqueue(): Promise<void> {
    const ego = this.context.ego();
    if (!ego?.targetFrame()) throw new Error('请先开启双目视觉并等待新鲜画面');
    const value = (id: string): string => document.querySelector<HTMLInputElement | HTMLSelectElement>(`#${id}`)!.value;
    if (!this.selected) throw new Error('请先选择任意标注目标');
    const selection = { clientId: ego.sessionId, targetId: this.selected.targetId };
    await this.context.request('/api/sim/contact', { selection, effector: value('touch-hand'), operation: value('touch-operation'),
      duration: Number(value('touch-duration')), amplitude: Number(value('touch-amplitude')) });
    this.status('触摸命令已排队 · 先停步，再生成接近 / 操作 / 撤回');
  }

  async execute(args: TouchArgs, current: () => boolean): Promise<void> {
    try { await this.executeCurrent(args, current); }
    catch (error) {
      if (current()) this.status(error instanceof Error ? error.message : '触摸失败');
      throw error;
    }
  }

  private async executeCurrent(args: TouchArgs, current: () => boolean): Promise<void> {
    this.cancel(); const revision = this.revision;
    this.contactSamples = 0; this.reachedSamples = 0;
    delete document.querySelector<HTMLElement>('#contact-live')!.dataset.tracking;
    await this.context.stabilize(current);
    if (!current() || revision !== this.revision) return;
    const seed = this.context.seed();
    const vrm = this.context.vrm(); if (!vrm) throw new Error('VRM 尚未加载');
    const name = args.effector ?? (args.hand === 'left' ? 'left_hand' : 'right_hand');
    const own = selfEffector(vrm, name);
    const started = await this.context.request<TouchJob>('/api/touch/start', { ...args, history: seed.motion, selfEffector: own });
    if (!current() || revision !== this.revision) { await this.context.request('/api/touch/cancel', { id: started.id }); return; }
    this.job = started.id; this.status('ARDY 正在生成完整触摸动作 · 持续复核视觉目标');
    try { await this.awaitMotion(started.id, seed, current, revision); }
    finally { if (this.job === started.id && !this.watching) this.cancel(); }
  }

  private async awaitMotion(id: string, seed: MotionSeed, current: () => boolean, revision: number): Promise<void> {
    const deadline = performance.now()+90000;
    while (current() && revision === this.revision) {
      if (performance.now() > deadline) { await this.context.request('/api/touch/cancel', { id }); throw new Error('触摸生成超过 90 秒'); }
      const job = await this.context.request<TouchJob>(`/api/touch/jobs/${id}`);
      if (!current() || revision !== this.revision) return;
      if (job.status === 'failed' || job.status === 'cancelled') throw new Error(job.error || '触摸生成已取消');
      if (job.status === 'ready' && job.motion) {
        const vrm = this.context.vrm(); if (!vrm) throw new Error('VRM 尚未加载');
        const fitted = fitTouchContact(vrm, this.context.root, job.motion, job.plan, seed.motion.frames);
        await this.context.request('/api/touch/validate', { id, motion: fitted.motion });
        const target = await this.verify(job.plan);
        if (!current() || revision !== this.revision) return;
        this.live = new LiveContact(vrm, job.plan, seed.motion.frames, this.context.obstacles()); this.live.observe(target);
        this.watching = job.plan; this.context.play(fitted.motion, seed);
        this.status(`接触播放 · ${job.plan.effector.name} · VRM 轨迹偏差 ${(fitted.error*100).toFixed(1)} cm · 双目 σ ${(job.plan.touch.target.sigmaMeters*100).toFixed(1)} cm`);
        return;
      }
      await new Promise<void>(resolve => window.setTimeout(resolve, 300));
    }
  }

  private async verify(plan: TouchPlan): Promise<TouchTarget> {
    return this.context.request<TouchTarget>('/api/touch/track', { id: plan.id });
  }

  private async watch(): Promise<void> {
    if (!this.watching || this.watchingBusy) return;
    this.watchingBusy = true; const plan = this.watching;
    try { const target = await this.verify(plan); if (plan === this.watching) this.live?.observe(target); }
    catch (error) {
      if (plan !== this.watching) return;
      const message = error instanceof Error ? error.message : '触摸目标丢失';
      this.cancel(); this.status(message); this.context.stop(message);
    } finally { this.watchingBusy = false; }
  }

  update(time: number, delta: number, playing: boolean): void {
    if (!playing || !this.live) return;
    const view = document.querySelector<HTMLElement>('#contact-live')!;
    try {
      const report = this.live.update(time, delta); if (!report) return;
      if (report.contactPhase) this.contactSamples++;
      if (report.observedContact) this.reachedSamples++;
      view.dataset.tracking = JSON.stringify({ ...report, contactSamples: this.contactSamples, reachedSamples: this.reachedSamples });
      view.textContent = `${report.contactPhase ? '接触阶段' : '接近 / 撤回'} · 有效接触 ${this.reachedSamples}/${this.contactSamples} 采样 · 当前法向间隙 ${(report.normalGapMeters*100).toFixed(1)} cm · 切向偏差 ${(report.tangentMeters*100).toFixed(1)} cm · 测距 σ ${(report.sigmaMeters*100).toFixed(1)} cm`;
    } catch (error) {
      const message = error instanceof Error ? error.message : '动态接触求解失败';
      this.cancel(); this.status(message); this.context.stop(message);
    }
  }

  private refresh(): void {
    const ego = this.context.ego(), vision = ego?.touchPeople();
    const select = document.querySelector<HTMLSelectElement>('#touch-person')!;
    const people = vision?.people ?? [], prior = select.value;
    select.replaceChildren(...people.map(person => new Option(`${person.id} · ${person.distance?.toFixed(2) ?? '?'} m`, person.id)));
    if (!people.length) select.add(new Option('等待双目人体检测', ''));
    if (people.some(person => person.id === prior)) select.value = prior;
    const frame = ego?.targetFrame();
    if (!frame || frame.frameId === this.rendered?.frameId) return;
    const image = new Image(), revision = ++this.imageRevision;
    image.onload = () => {
      if (revision !== this.imageRevision) return;
      this.rendered = frame;
      const context = document.querySelector<HTMLCanvasElement>('#touch-image')!.getContext('2d')!;
      context.drawImage(image, 0, 0); this.renderSelection();
      context.fillStyle = '#ffcb69';
      for (const person of vision?.frameId === frame.frameId ? people : []) for (const point of person.keypoints) if (point.confidence >= .6) {
        context.beginPath(); context.arc(point.pixel[0], point.pixel[1], 4, 0, Math.PI*2); context.fill();
      }
    };
    image.onerror = () => this.status('无法读取标注画面'); image.src = frame.image;
  }

  private async selectPixel(event: MouseEvent): Promise<void> {
    const frame = this.rendered, ego = this.context.ego(); if (!frame || !ego) throw new Error('没有有效检测帧');
    if (Date.now()/1000-frame.observedAt >= 1.5) throw new Error('标注画面已过期，请等待更新后重新选择');
    const canvas = event.currentTarget as HTMLCanvasElement, rect = canvas.getBoundingClientRect();
    const pixel = [(event.clientX-rect.left)*canvas.width/rect.width, (event.clientY-rect.top)*canvas.height/rect.height];
    const label = document.querySelector<HTMLInputElement>('#touch-label')!.value.trim();
    const target = await perceptionRequest<TouchTarget>('/api/touch/resolve', { clientId: ego.sessionId, pixel, frameId: frame.frameId, label });
    this.selectTarget(target);
  }

  selectTarget(target: TouchTarget): void {
    this.targets.set(target.targetId, target);
    if (this.targets.size > 16) this.targets.delete(this.targets.keys().next().value!);
    this.selected = target;
    const select = document.querySelector<HTMLSelectElement>('#touch-target')!;
    select.replaceChildren(...Array.from(this.targets.values(), item => new Option(
      `${item.label ?? item.text ?? item.anatomy?.part ?? '标注'} · ${item.targetId.slice(-6)}`, item.targetId)));
    select.value = target.targetId;
    this.renderSelection();
    this.status(`已选标注 · σ ${(target.sigmaMeters*100).toFixed(1)} cm · 使用下方统一操作执行`);
  }

  private async selectPart(): Promise<void> {
    const ego = this.context.ego();
    const personId = document.querySelector<HTMLSelectElement>('#touch-person')!.value;
    if (!ego || !personId) throw new Error('没有新鲜的人体部位标注');
    const bodyPart = document.querySelector<HTMLSelectElement>('#touch-part')!.value;
    this.selectTarget(await perceptionRequest<TouchTarget>('/api/touch/resolve', { clientId: ego.sessionId, personId, bodyPart }));
  }

  private renderSelection(): void {
    const canvas = document.querySelector<HTMLCanvasElement>('#touch-image')!;
    canvas.dataset.frame = JSON.stringify(this.rendered && { frameId: this.rendered.frameId, observedAt: this.rendered.observedAt });
    const view = document.querySelector<HTMLElement>('#touch-status')!;
    view.dataset.target = JSON.stringify(this.selected);
    const target = this.selected;
    if (!target?.requestedPixel || target.frameId !== this.rendered?.frameId) return;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('无法绘制视觉标注');
    const [x, y] = target.requestedPixel;
    context.strokeStyle = '#75e9ff'; context.lineWidth = 2;
    context.beginPath(); context.arc(x, y, 8, 0, Math.PI*2); context.stroke();
  }

  private status(message: string): void { document.querySelector('#touch-status')!.textContent = message; }
}

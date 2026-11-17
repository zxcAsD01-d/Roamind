import type { VisualObservation } from './perception';
import type { EffectorName } from './contact_rig';
import type { ComputerState } from './computer';

import { MemoryView } from './memory_view';
import { EngineView } from './engine_view';
import type { GoalStep, CharacterGoal, BehaviorMemory } from './character_types';
export type { GoalStep, CharacterGoal, BehaviorMemory } from './character_types';
type Context = { request: <T>(path: string, body: object) => Promise<T>; clientId: string;
  enableVision: () => void; suspendAutonomy: () => void; ready: () => boolean };
export type KernelState = { needs?: { energy: number; boredom: number; satiety: number; thirst: number; initiative: number;
  seekingRest: boolean; seekingFood: boolean; seekingWater: boolean }; intent?: { phase: string; drive: string; targetId: string | null };
  enabled: boolean; reason: string; consecutiveFailures: number; retryAfterSeconds: number };
const ACTIVE = new Set(['running', 'waiting']);
const intentLabels: Record<string, string> = { idle: '待机', 'explore-social': '探索与社交', 'search-resource': '探索寻找资源',
  navigate: '连续导航', approach: '接近目标', 'use-resource': '吃喝执行', interact: '物体交互',
  'body-transition': '姿态过渡', rest: '休息', recover: '恢复与换路', act: '执行动作' };
const labels: Record<string, string> = { social: '探索找人', waypoint: '移动到视觉位置', face: '面向',
  interact: '物体交互', body: '身体状态',
  action: '执行动作', wait: '自然等待', contact: '触碰所选标注', 'text-touch': '触碰视野文字',
  running: '执行中', waiting: '等待观测', completed: '完成', cancelled: '已取消', failed: '失败', pending: '待执行' };

export const characterPanel = `<section class="character-panel">
  <div class="panel-caption"><span>MISSION CONTROL</span><span>01</span></div><h1>下一步，去探索。</h1><p class="panel-intro">设定意图，让角色连续完成行动。</p>
  <p id="character-state" class="runtime-state" role="status">正在准备角色…</p>
  <label class="setting-row"><span>自主生活 · 探索与社交</span><input id="agent-autonomy" type="checkbox"><i class="switch"></i></label>
  <p id="agent-kernel-state" class="ardy-status" role="status">可开启环境驱动行为；直接指令会接管。</p>
  <div class="needs-grid" aria-label="角色内在状态">
    <label>精力 <output id="need-energy-value">80%</output><meter id="need-energy" min="0" max="100" low="30" high="68" optimum="85" value="80"></meter></label>
    <label>无聊 <output id="need-boredom-value">25%</output><meter id="need-boredom" min="0" max="100" value="25"></meter></label>
    <label>饱食度 <output id="need-satiety-value">78%</output><meter id="need-satiety" min="0" max="100" low="35" high="72" optimum="85" value="78"></meter></label>
    <label>口渴度 <output id="need-thirst-value">18%</output><meter id="need-thirst" min="0" max="100" low="22" high="60" optimum="0" value="18"></meter></label>
  </div><p id="need-policy" class="ardy-status">精力影响活动预算，无聊推动探索。</p>
  <div class="scene-buttons"><button id="body-stand" class="secondary-button">起身</button><button id="body-release" class="secondary-button">放回物品</button></div>
  <p id="computer-status" class="ardy-status" role="status">电脑房 · 识别键盘和座椅后，可在物体交互中玩电脑。</p>
  <label class="field-label" for="goal-kind">设定目标</label>
  <select id="goal-kind"><option value="social">探索环境，接近人物并挥手</option><option value="waypoint">前往视觉地图位置</option>
    <option value="face">转向指定方向</option><option value="action">执行动作</option><option value="contact">触碰所选标注</option></select>
  <div id="goal-fields"></div>
  <div class="scene-buttons"><button id="goal-start" class="primary-button">开始目标</button><button id="goal-cancel" class="secondary-button">停止 / 取消</button></div>
  <p id="goal-status" class="ardy-status" aria-live="polite">可设定目标；任务将保留视觉记忆。</p>
  <ol id="goal-steps" class="plan-timeline"></ol>
  <label class="setting-row"><span>自然待机短动作</span><input id="ambient-idle" type="checkbox" checked><i class="switch"></i></label>
  <details class="runtime-composer"><summary>组合任务 / LLM 指令</summary>
    <p class="ardy-status">按顺序执行，失败即停。位置使用视觉地图坐标（米）。</p>
    <textarea id="goal-json" aria-label="结构化角色任务" spellcheck="false">{"title":"寻找社交对象","steps":[{"kind":"social","timeout":600},{"kind":"wait","duration":2}]}</textarea>
    <button id="goal-json-run" class="secondary-button">执行组合任务</button>
    <p class="ardy-status">MCP：get_agent_context → submit_agent_intent；set_agent_autonomy 开启自主生活。游戏内文本输入适配器尚未接入。</p>
  </details>

</section>`;

export const memoryPanel = `  <section class="runtime-memory"><div class="panel-heading"><span>第一人称</span><span class="eyebrow">EGO VISION</span></div><div id="runtime-ego"></div><div class="panel-heading"><span>空间记忆</span><span class="eyebrow">VISUAL MAP</span></div>
    <p id="character-memory" class="ardy-status">等待双目定位；空白区域为未知。</p>
    <canvas id="memory-grid" width="320" height="200" aria-label="区域、边界与出口记忆图，点击已知区域设为目标"></canvas>
    <p class="ardy-status">底色：不同地板区域 · 实线：实测边界 · 灰虚线：未知<br>出口点：绿可通行 / 黄待确认 / 红受阻 · 连线：区域联通<br>点击已知区域设为目标；边界闭合仅描述已观测地板。</p>
    <details class="cloud-details"><summary>三维语义点云</summary><div id="runtime-cloud"></div></details>
  </section>`;

export class CharacterRuntime {
  goal: CharacterGoal | null = null;
  private memory: BehaviorMemory['spatialMemory'] | null = null;
  private memoryView = new MemoryView();
  private engineView = new EngineView();
  private submitting = false;
  private visionStamp = 0;
  private lastGoalRender = '';

  constructor(private context: Context) {
    this.bind('body-stand', () => this.submit({ title: '起身', steps: [{ kind: 'body', operation: 'stand' }] }));
    this.bind('body-release', () => this.submit({ title: '放回物品', steps: [{ kind: 'body', operation: 'release' }] }));
    this.bind('goal-start', () => this.submit({ title: labels[this.kind()], steps: [this.formStep()] }));
    this.bind('goal-cancel', () => this.cancel());
    this.bind('goal-json-run', () => this.submit(JSON.parse(this.value('goal-json')) as { title: string; steps: GoalStep[] }));
    document.querySelector('#goal-kind')!.addEventListener('change', () => this.renderFields());
    document.querySelector('#memory-grid')!.addEventListener('click', event => this.selectPoint(event as MouseEvent));
    document.querySelector<HTMLInputElement>('#agent-autonomy')!.addEventListener('change', event => {
      const input = event.currentTarget as HTMLInputElement;
      const enabled = input.checked;
      input.disabled = true;
      void this.setAutonomy(enabled).catch(error => {
        input.checked = !enabled;
        this.status(error instanceof Error ? error.message : '自主模式切换失败');
      }).finally(() => { input.disabled = false; });
    });
    this.renderFields();
    this.moveView('runtime-ego', ['ego-preview']);
    this.moveView('runtime-cloud', ['world-cloud', 'world-status', 'world-objects']);
    this.moveView('runtime-performance', ['power-toggle-row', 'power-status', 'performance-status']);
    this.moveView('runtime-actions', ['motion-table-panel', 'action-library-panel']);
    this.moveView('runtime-interactions', ['visual-contact-panel']);
    this.moveView('touch-ocr-selection', ['visual-text-panel']);
  }

  syncKernel(state: KernelState): void {
    if (state.needs) {
      for (const key of ['energy', 'boredom', 'satiety', 'thirst'] as const) {
        document.querySelector<HTMLMeterElement>(`#need-${key}`)!.value = state.needs[key];
        document.querySelector(`#need-${key}-value`)!.textContent = `${state.needs[key].toFixed(0)}%`;
      }
      document.querySelector('#need-policy')!.textContent = (state.needs.seekingWater ? '口渴：寻找饮水' : state.needs.seekingFood ? '饥饿：寻找食物'
        : state.needs.seekingRest ? '疲劳：寻找支撑面休息' : '探索与社交')+` · 活动意愿 ${(state.needs.initiative*100).toFixed(0)}%`;
    }
    const input = document.querySelector<HTMLInputElement>('#agent-autonomy')!;
    if (!input.disabled) input.checked = state.enabled;
    document.querySelector('#agent-kernel-state')!.textContent = state.reason
      +(state.intent ? ` · ${intentLabels[state.intent.phase] ?? state.intent.phase}` : '')
      +(state.retryAfterSeconds > 1 ? ` · ${state.retryAfterSeconds.toFixed(0)} 秒后重试` : '');
  }

  syncComputer(state: ComputerState | null): void {
    document.querySelector('#computer-status')!.textContent = state
      ? `${state.active ? '正在玩电脑' : state.completed ? '电脑游戏已结束' : '电脑游戏已停止'} · ${state.elapsed.toFixed(0)} 秒 · ${state.score} 分 · ${state.inputs} 次按键`
      : '电脑房 · 识别键盘和座椅后，可在物体交互中玩电脑。';
  }

  private async setAutonomy(enabled: boolean): Promise<void> {
    if (enabled && !this.context.ready()) throw new Error('角色和基础动画尚未就绪');
    this.context.suspendAutonomy();
    if (enabled) this.context.enableVision();
    this.syncKernel(await this.context.request<KernelState>('/api/agent/autonomy', { clientId: this.context.clientId, enabled }));
  }

  get active(): boolean { return !!this.goal && ACTIVE.has(this.goal.status); }

  get navigating(): boolean {
    const step = this.goal?.steps[this.goal.index];
    return this.active && !!step && (['social', 'waypoint'].includes(step.kind)
      || step.kind === 'interact' && ['approach', 'touch', 'tap', 'hold', 'stroke', 'sit', 'lie', 'pickup', 'eat', 'drink', 'play_computer'].includes(step.operation ?? ''));
  }

  sync(goal: CharacterGoal | null, behavior: BehaviorMemory, observation: VisualObservation | null,
    state: { playing: boolean; generating: boolean; activeCommand: string | null; phase: string }): void {
    this.goal = goal;
    if (this.active) { this.context.suspendAutonomy(); this.context.enableVision(); }
    document.querySelector<HTMLSelectElement>('#motion-speed')!.disabled = this.active;
    document.querySelector<HTMLInputElement>('#motion-scrub')!.disabled = this.active;
    this.renderGoal();
    this.memory = behavior.spatialMemory;
    this.memoryView.update(this.memory);
    this.engineView.update(behavior.engine, observation, state, this.active);
    const age = observation ? Date.now()/1000-observation.observedAt : Infinity;
    this.visionStamp = observation?.observedAt ?? 0;
    const control = this.active ? '目标控制' : state.activeCommand ? '直接控制' : '自由待机';
    const vision = age >= 1.5 ? '等待新视觉帧' : observation?.stereo.mapping ? '视觉定位在线' : '视觉定位丢失';
    document.querySelector('#character-state')!.textContent = `${control} · ${state.generating ? '准备动作' : state.playing ? state.phase : '已停稳'} · ${vision}`;
  }

  async submit(body: { title: string; steps: GoalStep[] }): Promise<void> {
    if (this.submitting) return;
    if (!this.context.ready()) throw new Error('角色和基础动画尚未就绪');
    this.submitting = true;
    try {
      this.context.suspendAutonomy();
      this.context.enableVision();
      this.goal = await this.context.request<CharacterGoal>('/api/goals/submit', { ...body, clientId: this.context.clientId });
      this.renderGoal();
    } finally { this.submitting = false; }
  }

  disconnected(message: string): void { this.engineView.disconnected(message); }

  private async cancel(): Promise<void> {
    this.context.suspendAutonomy();
    const result = await this.context.request<{ goal: CharacterGoal | null }>('/api/goals/cancel', { clientId: this.context.clientId });
    this.goal = result.goal;
    this.renderGoal();
    if (!this.goal) this.status('已停止直接控制');
  }

  private bind(id: string, task: () => Promise<void>): void {
    document.querySelector(`#${id}`)!.addEventListener('click', () => {
      void task().catch(error => this.status(error instanceof Error ? error.message : '角色操作失败'));
    });
  }

  private value(id: string): string { return document.querySelector<HTMLInputElement>(`#${id}`)!.value.trim(); }
  private kind(): string { return this.value('goal-kind'); }
  private status(message: string): void { document.querySelector('#goal-status')!.textContent = message; }

  private formStep(): GoalStep {
    const kind = this.kind();
    if (kind === 'waypoint') {
      const position = this.value('goal-position').split(/[,，\s]+/).map(Number);
      if (position.length !== 3 || !position.every(Number.isFinite)) throw new Error('请输入视觉地图坐标 x,y,z');
      return { kind, position };
    }
    if (kind === 'face') return { kind, heading: Number(this.value('goal-heading')) };
    if (kind === 'action') return { kind, action: this.value('goal-action'), duration: 3 };
    if (kind === 'contact') {
      const targetId = this.value('touch-target');
      if (!targetId) throw new Error('请先在交互面板选择任意标注目标');
      return { kind, targetId, effector: this.value('touch-hand') as EffectorName,
        operation: this.value('touch-operation'), duration: Number(this.value('touch-duration')),
        amplitude: Number(this.value('touch-amplitude')) };
    }
    return { kind, timeout: 600 };
  }

  private renderFields(): void {
    const fields: Record<string, string> = {
      social: '<p class="ardy-status">记住已确认区域与出口 → 连续探索新路线 → 保持同一人物目标 → 接近、停稳、挥手。</p>',
      waypoint: '<label class="field-label" for="goal-position">视觉位置 x,y,z（可点击下方地图）</label><input id="goal-position" class="scene-input" placeholder="0, 0, 1">',
      face: '<label class="field-label" for="goal-heading">自身控制坐标朝向（度）</label><input id="goal-heading" class="scene-input" type="number" value="90">',
      action: '<select id="goal-action"><option value="wave">挥手</option><option value="raise">抬手</option><option value="point">指向</option><option value="crouch">蹲下</option></select>',
      contact: '<p class="ardy-status">使用交互面板当前选中的任意标注位置，以及执行部位、操作和停留时间。</p>' };
    document.querySelector('#goal-fields')!.innerHTML = fields[this.kind()];
  }

  private renderGoal(): void {
    const status = document.querySelector<HTMLElement>('#goal-status')!;
    const serialized = JSON.stringify(this.goal);
    if (serialized === this.lastGoalRender) return;
    this.lastGoalRender = serialized;
    status.dataset.goal = serialized;
    if (!this.goal) return;
    this.status(`${this.goal.title} · ${labels[this.goal.status] ?? this.goal.status} · ${this.goal.detail}`);
    document.querySelector('#runtime-events')!.replaceChildren(...this.goal.events.slice(-12).map(event => {
      const row = document.createElement('li');
      row.textContent = `${new Date(event.at*1000).toLocaleTimeString()} · ${labels[event.status] ?? event.status} · ${event.detail}`;
      return row;
    }));
    document.querySelector('#goal-steps')!.replaceChildren(...this.goal.steps.map(step => {
      const row = document.createElement('li');
      row.textContent = `${labels[step.kind]} · ${labels[step.status ?? 'pending'] ?? step.status}`;
      row.dataset.status = step.status; return row;
    }));
  }

  private selectPoint(event: MouseEvent): void {
    if (!this.memory?.position || !this.visionStamp) return;
    const [x, z] = this.memoryView.positionAt(event);
    document.querySelector<HTMLSelectElement>('#goal-kind')!.value = 'waypoint'; this.renderFields();
    document.querySelector<HTMLInputElement>('#goal-position')!.value = `${x.toFixed(2)}, 0, ${z.toFixed(2)}`;
    this.status('已选视觉位置；开始目标前会检查是否存在已知可通行路径。');
  }

  private moveView(id: string, children: string[]): void {
    const parent = document.querySelector(`#${id}`);
    if (!parent) throw new Error(`运行台缺少容器：${id}`);
    for (const child of children) {
      const element = document.querySelector(`#${child}`);
      if (!element) throw new Error(`运行台缺少组件：${child}`);
      parent.append(element);
    }
  }
}

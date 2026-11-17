import type { VisualObservation } from './perception';

export const behaviorPanel = `
  <details class="behavior-panel" open>
    <summary>主动探索 / 寻找社交</summary>
    <label class="setting-row"><span>启用找人社交脚本</span><input id="autonomy-toggle" type="checkbox"><i class="switch"></i></label>
    <div class="scene-buttons"><button id="autonomy-reset" class="secondary-button">重新探索 / 清除记忆</button></div>
    <p class="ardy-status">出口优先持续探索 → 找人接近 → 挥手。未知深度继续走，受阻回头换路。</p>
    <p id="autonomy-status" class="ardy-status">关闭 · 社交条件：0.9–1.7 米 / 面向 ±12° / 连续确认</p>
  </details>`;

type SimulationState = { playing: boolean; generating: boolean; following: boolean;
  buffered: number; brakeIn: number | null; visualCommandId: string | null; handoffReady?: boolean };
type AutonomyCommand = { id: string; kind: 'visual'; arguments: Record<string, string | number | boolean | number[] | number[][] | undefined> };
type DecisionResponse = { busy: boolean; completed?: boolean; reason?: string; decision?: { action: string; duration: number; intent?: string;
  reason?: string; mission?: { phase: string; ready: boolean; confirmationFrames: number };
  features: Record<string, number> }; command?: AutonomyCommand };
type Hooks = { request: (body: object) => Promise<DecisionResponse>; reset: () => Promise<void>;
  command: (command: AutonomyCommand) => void; status: (message: string) => void; clientId: string;
  vision: () => VisualObservation | null; enableVision: () => void; hold: (reason: string) => void };

export class AutonomyController {
  private enabled = false;
  private finished = false;
  private busy = false;
  private timer: number | null = null;
  private timerDue = 0;
  private activeAction = '';
  private awaitingCommand: string | null = null;
  private nextDecisionAt = 0;
  private lastVisionAt = 0;
  private playback: SimulationState = { playing: false, generating: false, following: false, buffered: 0, brakeIn: null, visualCommandId: null };

  constructor(private hooks: Hooks) {
    document.querySelector<HTMLInputElement>('#autonomy-toggle')!.addEventListener('change', event => {
      this.enabled = (event.target as HTMLInputElement).checked;
      this.finished = false;
      if (this.enabled) {
        this.hooks.enableVision();
        this.hooks.status('脚本已开启 · 持续规划探索区域，直到社交条件达成');
        this.schedule(0);
      } else {
        this.cancel();
        this.hooks.hold('自主策略已关闭');
        this.hooks.status('已关闭 · 学习记忆保留');
      }
    });
    document.querySelector('#autonomy-reset')!.addEventListener('click', () => {
      this.hooks.hold('重新探索');
      void this.hooks.reset().then(() => {
        this.finished = false; this.lastVisionAt = 0; this.awaitingCommand = null;
        this.hooks.status('记忆已清除 · 重新规划找人社交');
        if (this.enabled) this.schedule(450);
      })
        .catch(error => this.hooks.status(error instanceof Error ? error.message : '清除学习记忆失败'));
    });
  }

  get active(): boolean { return this.enabled && !this.finished; }

  suspend(): void {
    if (!this.enabled) return;
    this.enabled = false; this.activeAction = ''; this.awaitingCommand = null; this.cancel();
    document.querySelector<HTMLInputElement>('#autonomy-toggle')!.checked = false;
    this.hooks.status('已让出控制权 · 视觉记忆保留');
  }

  update(state: SimulationState): void {
    this.playback = state;
    if (state.visualCommandId === this.awaitingCommand) this.awaitingCommand = null;
    if (!this.enabled || this.busy || !this.canDecide()) return;
    if (!this.hooks.vision()) return;
    this.schedule(0);
  }

  private canDecide(): boolean {
    const state = this.playback;
    const continuation = state.playing && state.visualCommandId !== null
      && ((this.activeAction === 'forward' && ((state.brakeIn !== null && state.brakeIn > 0 && state.brakeIn <= 1.6)
        || state.buffered <= .25)) || state.handoffReady === true) && state.buffered > 0;
    return !this.finished && !this.awaitingCommand && performance.now() >= this.nextDecisionAt
      && !state.generating && !state.following && (continuation || (!state.playing && state.visualCommandId === null));
  }

  private schedule(delay: number): void {
    const due = performance.now() + delay;
    if (this.timer !== null && this.timerDue <= due) return;
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.timerDue = due;
    this.timer = window.setTimeout(() => { this.timer = null; this.timerDue = 0; void this.decide(); }, delay);
  }

  private cancel(): void {
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.timer = null; this.timerDue = 0;
  }

  private async decide(): Promise<void> {
    if (!this.enabled || this.busy || !this.canDecide()) return;
    this.busy = true;
    try {
      const vision = this.hooks.vision();
      if (!vision) { this.hooks.status('等待第一人称视觉帧'); return; }
      if (vision.observedAt <= this.lastVisionAt) return;
      this.lastVisionAt = vision.observedAt;
      const result = await this.hooks.request({ clientId: this.hooks.clientId, vision,
        afterCommand: this.playback.visualCommandId });
      if (!this.enabled) return;
      if (result.busy) {
        this.hooks.status(result.reason ?? '策略等待当前动作完成');
      } else if (result.completed) {
        this.finished = true; this.activeAction = ''; this.cancel();
        this.hooks.status(result.reason ?? '社交条件已达成 · 已停步，等待交互');
      } else if (result.command) {
        if (!this.enabled) return;
        const decision = result.decision;
        this.awaitingCommand = result.command.id;
        this.activeAction = decision?.action ?? '';
        this.hooks.status(decision?.reason ?? decision?.intent ?? '正在规划探索区域');
        this.hooks.command(result.command);
      }
    } catch (error) {
      this.hooks.status(error instanceof Error ? error.message : '自主策略失败');
    } finally {
      this.busy = false;
      this.nextDecisionAt = performance.now()+400;
      if (this.enabled && !this.finished) this.schedule(450);
    }
  }
}

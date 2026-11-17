import type { VisualObservation } from './perception';

type Entity = { id: string; visible: boolean; ageSeconds: number; distance: number | null;
  confidence: number; velocity: number[]; samples: number; worldPosition: number[] | null };
type Candidate = { id: string; utility: number; components: Record<string, number> };
export type EngineSnapshot = { version: number; phase: string; intent: string; reason?: string;
  world: { observedAt: number; self: { localized: boolean }; entities: Entity[] };
  arbitration: { targetId: string | null; heldSeconds: number; reason: string;
    switches: number; candidates: Candidate[] };
  socialGraph: { edges: { source: string; target: string; distance: number }[];
    groups: { id: string; members: string[]; formation: string }[] };
  execution: { status: string; action: string | null; observedAt: number; detail: string };
  events: { at: number; kind: string; detail: string }[] };
type PlaybackState = { playing: boolean; generating: boolean; phase: string };
const phases: Record<string, string> = { Explore: '持续探索', Observe: '观察目标', Approach: '接近目标',
  Join: '确认社交条件', Interact: '执行互动', Maintain: '互动已完成', Disengage: '离开交互', Recover: '恢复路线' };

export const enginePanel = `<section class="engine-panel">
  <div class="panel-heading"><span>分层行为引擎</span><span class="eyebrow">LIVE STATE</span></div>
  <div class="engine-current"><span id="engine-phase" class="phase-badge">等待观测</span><p id="engine-reason">视觉记忆准备中</p></div>
  <div class="engine-flow" aria-label="行为执行链"><span>感知记忆</span><b>→</b><span>目标仲裁</span><b>→</b><span>动作执行</span></div>
  <div class="engine-metrics"><div><strong id="engine-entities">0</strong><span>人物记忆</span></div><div><strong id="engine-groups">0</strong><span>空间聚集</span></div><div><strong id="engine-hold">—</strong><span>目标持续</span></div></div>
  <div class="panel-heading"><span>人物与目标</span><span id="engine-target" class="eyebrow">NO TARGET</span></div>
  <div id="engine-people" class="entity-list"><p class="empty-state">人物会在视觉识别后进入记忆。</p></div>
  <details class="engine-details"><summary>候选效用与执行反馈</summary><p class="ardy-status">效用为启发式评分。空间聚集尚未验证交谈关系。</p><div id="engine-candidates"></div><p id="engine-feedback" class="ardy-status">尚无执行反馈</p></details>
  <details class="engine-details"><summary>行为转换记录</summary><ol id="engine-events" class="plan-timeline"></ol></details>
</section>`;

function text(id: string, value: string): void {
  const element = document.getElementById(id);
  if (element && element.textContent !== value) element.textContent = value;
}

export class EngineView {
  private signature = '';
  private rows = '';
  private eventSignature = '';
  private staleTimer = 0;

  update(engine: EngineSnapshot | undefined, observation: VisualObservation | null,
    playback: PlaybackState, active: boolean): void {
    window.clearTimeout(this.staleTimer);
    this.staleTimer = window.setTimeout(() => this.disconnected('运行状态超过 5 秒未更新'), 5000);
    const fresh = !!observation && Date.now()/1000-observation.observedAt < 1.5;
    text('workspace-connection', fresh ? '服务已连接 · 视觉在线' : '服务已连接 · 等待视觉');
    document.querySelector('#workspace-connection')!.setAttribute('data-state', fresh ? 'online' : 'waiting');
    const activity = playback.generating ? '准备动作' : playback.playing ? '动作播放中' : active ? '规划行动' : '自由待机';
    text('stage-activity', activity);
    if (!engine) return;
    text('engine-phase', phases[engine.phase] ?? engine.phase);
    document.querySelector('#engine-phase')!.setAttribute('data-phase', engine.phase);
    text('engine-reason', engine.reason ?? engine.arbitration.reason);
    text('engine-entities', String(engine.world.entities.length));
    text('engine-groups', String(engine.socialGraph.groups.length));
    text('engine-hold', engine.arbitration.targetId ? `${engine.arbitration.heldSeconds.toFixed(1)}s` : '—');
    text('engine-target', engine.arbitration.targetId ?? 'NO TARGET');
    const feedback = engine.execution;
    text('engine-feedback', feedback.action ? `${feedback.action} · ${feedback.status} · ${feedback.detail}` : '尚无执行反馈');
    this.renderPeople(engine.world.entities, engine.arbitration.targetId);
    this.renderCandidates(engine.arbitration.candidates);
    this.renderEvents(engine.events);
  }

  disconnected(message: string): void {
    text('workspace-connection', '连接中断');
    document.querySelector('#workspace-connection')!.setAttribute('data-state', 'offline');
    text('stage-activity', '等待服务');
    text('engine-feedback', message);
  }

  private renderPeople(entities: Entity[], target: string | null): void {
    const signature = JSON.stringify(entities.map(row => [row.id, row.visible, row.distance?.toFixed(1), row.samples, row.ageSeconds.toFixed(0)]))+target;
    if (signature === this.rows) return;
    this.rows = signature;
    const rows = entities.map(row => {
      const card = document.createElement('article'); card.className = 'entity-card';
      card.dataset.selected = String(row.id === target);
      const name = document.createElement('strong'); name.textContent = row.id;
      const state = document.createElement('span'); state.className = 'entity-state';
      state.textContent = row.visible ? '可见' : `遮挡 ${row.ageSeconds.toFixed(0)}s`;
      const metric = document.createElement('p');
      metric.textContent = `${row.distance === null ? '距离未知' : row.distance.toFixed(1)+' m'} · ${row.samples} 帧证据${row.id === target ? ' · 当前目标' : ''}`;
      card.append(name, state, metric); return card;
    });
    if (!rows.length) {
      const empty = document.createElement('p'); empty.className = 'empty-state'; empty.textContent = '视野中暂无人物，继续收集环境信息。';
      document.querySelector('#engine-people')!.replaceChildren(empty);
    } else document.querySelector('#engine-people')!.replaceChildren(...rows);
  }

  private renderCandidates(candidates: Candidate[]): void {
    const signature = JSON.stringify(candidates);
    if (signature === this.signature) return;
    this.signature = signature;
    document.querySelector('#engine-candidates')!.replaceChildren(...candidates.map(row => {
      const line = document.createElement('div'); line.className = 'utility-row';
      const label = document.createElement('span'); label.textContent = row.id;
      const value = document.createElement('strong'); value.textContent = row.utility.toFixed(2);
      line.title = Object.entries(row.components).map(([key, score]) => `${key}: ${score.toFixed(2)}`).join('\n');
      line.append(label, value); return line;
    }));
  }

  private renderEvents(events: EngineSnapshot['events']): void {
    const signature = JSON.stringify(events);
    if (signature === this.eventSignature) return;
    this.eventSignature = signature;
    document.querySelector('#engine-events')!.replaceChildren(...events.slice(-12).reverse().map(event => {
      const line = document.createElement('li');
      line.textContent = `${new Date(event.at*1000).toLocaleTimeString()} · ${event.detail}`; return line;
    }));
  }
}

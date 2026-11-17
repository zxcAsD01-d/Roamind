import { characterPanel, memoryPanel } from './character';
import { objectInteractionPanel } from './object_interactions';
import { targetPanel } from './targets';
import { behaviorPanel } from './behavior';
import { perceptionPanel } from './perception';
import { touchPanel } from './touch';
import { ocrPanel } from './ocr';
import { actionLibraryPanel } from './action_library';
import { enginePanel } from './engine_view';

export const workspaceMarkup = `
  <header class="topbar">
    <a class="brand" href="#" aria-label="具身行为实验室首页"><span class="brand-mark">M</span><span>具身行为实验室<small>SOCIAL MOTION LAB</small></span></a>
    <div class="top-meta"><span class="live-dot"></span><span id="workspace-connection" role="status">正在连接</span><span class="divider"></span><span>LOCAL WORKSPACE</span></div>
    <button class="icon-button" id="reset-all" title="重置姿势与视角" aria-label="重置姿势与视角"><span>↺</span></button>
  </header>
  <main class="workspace">
    <aside class="sidebar left-panel">
      ${characterPanel}
      <div class="panel-caption"><span>WORKSPACE</span><span>01</span></div>
      <details class="research-controls"><summary>场景与感知设置</summary>
      <div class="section-head"><div><p class="eyebrow">LIBRARY</p><h1>模型库</h1></div><span class="count">03</span></div>
      <div class="model-list" id="model-list"></div>
      <div class="panel-rule"></div>
      <section class="simulation-panel">
        <div class="section-head"><div><p class="eyebrow">HOUSE SIMULATION</p><h2>四室两厅三卫一厨</h2></div></div>
        <label class="setting-row"><span>房屋仿真 / 碰撞</span><input id="house-toggle" type="checkbox" checked><i class="switch"></i></label>
        <label class="setting-row"><span>剖切观察</span><input id="cutaway-toggle" type="checkbox" checked><i class="switch"></i></label>
        <label class="setting-row"><span>碰撞体辅助</span><input id="collider-toggle" type="checkbox"><i class="switch"></i></label>
        <label class="field-label" for="destination-select">移动到房间</label>
        <select id="destination-select"><option>读取房屋…</option></select>
        <label class="field-label" for="route-via">途经房间（逗号分隔，可选）</label>
        <input id="route-via" class="scene-input" placeholder="走廊, 餐厅">
        <div class="scene-buttons"><button id="preview-route" class="secondary-button">预览路线</button><button id="execute-route" class="primary-button">移动到这里</button></div>
        <div class="scene-buttons"><button id="reset-house" class="secondary-button">回到出生点</button></div>
        <details class="timeline-editor"><summary>含时间动作规划</summary>
          <textarea id="plan-editor" aria-label="含时间动作规划" spellcheck="false">[{"action":"walk","start":0,"end":14,"destination":"dining"},{"action":"wave","start":14,"end":18}]</textarea>
          <div class="scene-buttons"><button id="preview-plan" class="secondary-button">校验时间表</button><button id="execute-plan" class="primary-button">执行时间表</button></div>
        </details>
        <ol id="plan-timeline" class="plan-timeline"></ol>
        <p id="simulation-status" class="ardy-status">正在连接仿真环境…</p>
        <label id="power-toggle-row" class="setting-row"><span>节能模式</span><input id="power-toggle" type="checkbox" checked><i class="switch"></i></label>
        <p id="power-status" class="ardy-status">活动 30 FPS · 静止 15 FPS · 感知按需降频</p>
        <p id="performance-status" class="ardy-status"></p>
      </section>
      ${targetPanel}
      ${behaviorPanel}
      <div class="panel-rule"></div>
      ${perceptionPanel}
      ${touchPanel}
      ${ocrPanel}
      <div class="panel-rule"></div>
      <div class="section-head compact"><div><p class="eyebrow">VIEWPORT</p><h2>观察设置</h2></div></div>
      <label class="setting-row"><span>网格地面</span><input id="grid-toggle" type="checkbox" checked><i class="switch"></i></label>
      <label class="setting-row"><span>坐标辅助</span><input id="axes-toggle" type="checkbox"><i class="switch"></i></label>
      <div class="setting-row"><span>背景</span><div class="swatches"><button class="swatch selected" data-bg="#191e20" aria-label="深色背景"></button><button class="swatch light" data-bg="#d9ddda" aria-label="浅色背景"></button><button class="swatch green" data-bg="#273630" aria-label="绿色背景"></button></div></div>
      <div class="left-footer"><span class="status-led"></span><span id="load-status">正在准备场景</span></div>
      </details>
    </aside>
    <section class="stage" aria-label="3D 模型视口">
      <div class="stage-top"><div class="scene-heading"><span class="eyebrow">LIVE SIMULATION / 01</span><span class="stage-model" id="stage-model">WOLFERIA</span></div><span class="stage-tag" id="stage-activity">等待角色就绪</span></div>
      <div id="viewport"></div>
      <div class="stage-bottom"><span class="hint"><kbd>拖拽</kbd>旋转视角 <span class="hint-sep">/</span> <kbd>滚轮</kbd>缩放</span><div class="camera-tools"><button id="frame-house" class="text-button">全屋视角</button><button id="frame-model" class="text-button">聚焦角色 <span>⌖</span></button></div></div>
      <div id="load-overlay" class="load-overlay"><div class="spinner"></div><span>正在加载模型</span><small>首次载入可能需要一点时间</small></div>
      <div id="error-toast" class="error-toast" role="alert"></div>
    </section>
    <aside class="sidebar right-panel" aria-label="运行状态与工具">
      <div class="inspector-header"><div><p class="eyebrow">LIVE INSPECTOR</p><h2>观察与运行</h2></div><span class="inspector-badge">视觉驱动</span></div>
      <div class="inspector-tabs" role="tablist" aria-label="观察面板">
        <button id="tab-world" role="tab" aria-selected="true" aria-controls="panel-world" data-panel="world">世界记忆</button>
        <button id="tab-social" role="tab" aria-selected="false" aria-controls="panel-social" data-panel="social" tabindex="-1">行为决策</button>
        <button id="tab-motion" role="tab" aria-selected="false" aria-controls="panel-motion" data-panel="motion" tabindex="-1">动作工具</button>
      </div>
      <section id="panel-world" class="inspector-panel" role="tabpanel" aria-labelledby="tab-world">${memoryPanel}</section>
      <section id="panel-social" class="inspector-panel" role="tabpanel" aria-labelledby="tab-social" hidden>${enginePanel}${objectInteractionPanel}
        <div class="panel-heading"><span>任务事件</span><span class="eyebrow">EVENT LOG</span></div><ol id="runtime-events" class="plan-timeline"><li>等待目标</li></ol>
      </section>
      <section id="panel-motion" class="inspector-panel" role="tabpanel" aria-labelledby="tab-motion" hidden>
      <section class="runtime-inspector"><p class="eyebrow">EXECUTION</p><h2>执行与能力</h2>
        <p class="ardy-status">原生步态 / 转身 · 流明上身叠加 · 特殊动作按需生成</p>
        <div id="runtime-performance"></div><div id="runtime-actions"></div>
        <details class="runtime-interactions"><summary>人物与文字互动</summary><div id="runtime-interactions"></div></details>

      </section>
      <details class="research-controls"><summary>动作生成、轨迹与关节调试</summary>
      <section class="action-panel">
        <div class="section-head"><div><p class="eyebrow">ARDY MOTION</p><h2>动作播放</h2></div><span class="count" id="motion-count">--</span></div>
        <label class="field-label" for="motion-select">动作片段</label>
        <select id="motion-select"><option>读取本地动作…</option></select>
        <div class="motion-transport">
          <button id="play-motion" class="primary-button transport-play" title="播放或暂停" aria-label="播放或暂停">▶</button>
          <button id="stop-motion" class="mini-icon" title="停止并回到首帧" aria-label="停止并回到首帧">■</button>
          <label class="loop-control"><input id="motion-loop" type="checkbox" checked><span>循环</span></label>
          <select id="motion-speed" class="speed-select" aria-label="播放速度"><option value="0.5">0.5×</option><option value="0.75">0.75×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select>
        </div>
        <input id="motion-scrub" class="motion-scrub" type="range" min="0" max="1000" value="0" aria-label="动作时间轴">
        <div class="motion-time"><span id="motion-time">0.0 s</span><span id="motion-duration-label">0.0 s</span></div>
        <div class="panel-rule"></div>
        <label class="field-label" for="prompt-select">自回归提示词</label>
        <select id="prompt-select"><option>连接 ARDY 服务以读取缓存</option></select>
        <div class="generate-row"><label class="duration-field"><input id="generation-duration" type="number" min="1" max="60" step="1" value="24" aria-label="生成时长秒数"><span>秒</span></label><button id="generate-motion" class="secondary-button generate-button" title="从当前播放位置续生成；更换提示词后再次点击可连贯切换">续生成 / 切换动作</button></div>
        <p id="ardy-status" class="ardy-status">连接 ARDY 服务…</p>
        ${actionLibraryPanel}
        <details class="trajectory-editor" open>
          <summary>根节点轨迹 <span id="trajectory-count">03 点</span></summary>
          <div class="trajectory-grid trajectory-head"><span>秒</span><span>X</span><span>Y</span><span>Z</span><span>°</span><span></span></div>
          <div id="trajectory-rows" class="trajectory-rows"></div>
          <div class="trajectory-actions"><button id="add-trajectory-point" class="mini-icon" title="添加轨迹点" aria-label="添加轨迹点">＋</button><button id="generate-trajectory" class="secondary-button trajectory-generate">按轨迹生成并接播</button></div>
          <p class="trajectory-note">坐标单位：米；朝向：度</p>
        </details>
      </section>
      <div class="panel-rule action-divider"></div>
      <div class="section-head"><div><p class="eyebrow">HUMANOID RIG</p><h1>姿势控制</h1></div><button id="reset-joint" class="mini-icon" title="重置所选关节" aria-label="重置所选关节">↺</button></div>
      <label class="field-label" for="joint-select">选择关节</label>
      <select id="joint-select"></select>
      <div class="joint-diagram"><div class="joint-mark">✣</div><div><strong id="joint-name">髋部</strong><span>局部旋转 · 欧拉角</span></div></div>
      <div class="axis-controls" id="axis-controls"></div>
      <div class="angle-readout"><span>当前角度</span><strong id="angle-readout">X 0°　Y 0°　Z 0°</strong></div>
      <div class="panel-rule"></div>
      <div class="section-head compact"><div><p class="eyebrow">POSE</p><h2>姿势快照</h2></div><span class="count" id="snapshot-count">00</span></div>
      <div class="snapshot-actions"><button id="save-pose" class="primary-button"><span>＋</span> 保存当前姿势</button><button id="reset-pose" class="secondary-button" title="恢复模型初始姿势">恢复初始姿势</button></div>
      <div id="snapshots" class="snapshots"><p class="empty-state">调整关节后保存快照</p></div>
      <div class="right-footer"><span>旋转范围</span><span>−180° 至 180°</span></div>
      </details>
      </section>
    </aside>
  </main>
`;

export function bindWorkspaceTabs(): void {
  const tabs = [...document.querySelectorAll<HTMLButtonElement>('[role="tab"][data-panel]')];
  const activate = (selected: HTMLButtonElement): void => {
    for (const tab of tabs) {
      const active = tab === selected;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      document.querySelector<HTMLElement>(`#panel-${tab.dataset.panel}`)!.hidden = !active;
    }
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length-1
        : (index+(event.key === 'ArrowRight' ? 1 : -1)+tabs.length)%tabs.length;
      activate(tabs[next]); tabs[next].focus();
    });
  });
}

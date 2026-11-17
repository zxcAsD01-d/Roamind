async page => {
  const durationSeconds = 180;
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#ambient-idle').uncheck();
  await page.waitForFunction(() => document.querySelector('#ego-toggle')?.checked);
  await page.waitForFunction(() => JSON.parse(document.querySelector('#depth-status').dataset.result || '{}').mapping);
  await page.waitForTimeout(3000);
  return page.evaluate(async durationSeconds => {
    const clientId = document.querySelector('#viewport').dataset.clientId;
    const post = async (path, args = {}) => {
      const response = await fetch('http://127.0.0.1:8100'+path, { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...args, clientId }) });
      const result = await response.json();
      if (!response.ok) throw new Error(path+': '+JSON.stringify(result));
      return result;
    };
    const probe = window.hierarchicalProbe = { clientId, samples: [], states: [], commands: {}, errors: [], done: false };
    const started = performance.now();
    const sampleTimer = setInterval(() => {
      const root = document.querySelector('#viewport').dataset;
      const depth = JSON.parse(document.querySelector('#depth-status').dataset.result || '{}');
      probe.samples.push({ seconds: (performance.now()-started)/1000, x: Number(root.rootX), z: Number(root.rootZ),
        phase: root.phase, tracking: depth.mapping, stereoMs: depth.inferenceMs, exits: depth.exitEvidence });
    }, 200);
    let polling = false;
    const stateTimer = setInterval(async () => {
      if (polling) return;
      polling = true;
      try {
        const state = await post('/api/goals/state');
        probe.states.push({ seconds: (performance.now()-started)/1000, ...state });
        const response = await fetch('http://127.0.0.1:8100/api/sim/state');
        if (!response.ok) throw new Error('sim HTTP '+response.status);
        const sim = await response.json();
        const session = sim.sessions.find(row => row.id === clientId);
        probe.performance ??= [];
        if (session) probe.performance.push({seconds: (performance.now()-started)/1000, fps: session.state.fps, active: session.state.playing});
        sim.commands.filter(command => command.clientId === clientId).forEach(command => { probe.commands[command.id] = command; });
      } catch (error) { probe.errors.push(String(error)); }
      finally { polling = false; }
    }, 2000);
    try {
      probe.goal = await post('/api/goals/submit', { title: '分层引擎：连续探索与执行反馈实测', steps: [{ kind: 'social', timeout: durationSeconds+120 }] });
    } catch (error) { clearInterval(sampleTimer); clearInterval(stateTimer); throw error; }
    setTimeout(async () => {
      clearInterval(sampleTimer); clearInterval(stateTimer);
      try {
        probe.final = await post('/api/goals/state');
        probe.cleanup = await post('/api/goals/cancel');
        probe.ui = { phase: document.querySelector('#engine-phase').textContent, connection: document.querySelector('#workspace-connection').textContent, duplicateIds: [...document.querySelectorAll('[id]')].map(node => node.id).filter((id, index, ids) => ids.indexOf(id) !== index) };
        probe.powerSaving = document.querySelector('#power-toggle').checked;
        probe.microphone = !document.querySelector('#mic-stop').disabled;
      } catch (error) { probe.errors.push(String(error)); }
      finally { probe.done = true; }
    }, durationSeconds*1000);
    return { clientId, recordingSeconds: durationSeconds };
  }, durationSeconds);
}

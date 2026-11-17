async page => {
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.waitForFunction(() => JSON.parse(document.querySelector('#depth-status').dataset.result || '{}').mapping);
  await page.locator('#ambient-idle').uncheck();
  return page.evaluate(async () => {
    const clientId = document.querySelector('#viewport').dataset.clientId;
    const request = async (path, args = {}) => {
      const response = await fetch('http://127.0.0.1:8100' + path, { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...args, clientId }) });
      const value = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(value));
      return value;
    };
    const probe = window.navigationFlowProbe = { samples: [], commands: {}, errors: [], done: false };
    const started = performance.now();
    const sampleTimer = setInterval(() => {
      const root = document.querySelector('#viewport').dataset;
      const depth = JSON.parse(document.querySelector('#depth-status').dataset.result || '{}');
      const metrics = JSON.parse(document.querySelector('#performance-status').dataset.metrics || '{}');
      probe.samples.push({ seconds: (performance.now()-started)/1000, x: Number(root.rootX), z: Number(root.rootZ),
        phase: root.phase, time: Number(root.motionTime), buffer: Number(root.bufferSeconds),
        mode: root.motionMode, underflows: Number(root.underflows), tracking: depth.mapping,
        fps: metrics.fps, powerSaving: metrics.powerSaving, stereoMs: depth.inferenceMs });
    }, 100);
    let polling = false;
    const commandTimer = setInterval(async () => {
      if (polling) return;
      polling = true;
      try {
        const response = await fetch('http://127.0.0.1:8100/api/sim/state');
        if (!response.ok) throw new Error(`state HTTP ${response.status}`);
        const state = await response.json();
        state.commands.filter(command => command.clientId === clientId)
          .forEach(command => { probe.commands[command.id] = command; });
      } catch (error) { probe.errors.push(String(error)); }
      finally { polling = false; }
    }, 500);
    try {
      probe.goal = await request('/api/goals/submit', { title: '连续路线执行实测', steps: [{ kind: 'social', timeout: 150 }] });
    } catch (error) { clearInterval(sampleTimer); clearInterval(commandTimer); throw error; }
    setTimeout(async () => {
      clearInterval(sampleTimer); clearInterval(commandTimer);
      try {
        probe.final = await request('/api/goals/state');
        probe.cleanup = await request('/api/goals/cancel');
      } catch (error) { probe.errors.push(String(error)); }
      finally { probe.done = true; }
    }, 90000);
    return { clientId, recordingSeconds: 90 };
  });
}

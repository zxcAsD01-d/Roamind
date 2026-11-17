async page => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#power-toggle').check();
  await page.locator('#ego-toggle').evaluate(element => {
    element.checked = true; element.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.waitForFunction(() => JSON.parse(document.querySelector('#depth-status').dataset.result || '{}').mapping,
    null, { timeout: 30000 });
  return page.evaluate(async () => {
    const clientId = document.querySelector('#viewport').dataset.clientId;
    const nativeFetch = window.fetch.bind(window), started = performance.now();
    const probe = window.stabilityProbe = {
      clientId, started, samples: [], syncs: [], reports: [], longTasks: [], errors: [], done: false,
    };
    const observer = new PerformanceObserver(list => {
      for (const entry of list.getEntries()) {
        if (probe.longTasks.length < 512) probe.longTasks.push({ at: entry.startTime-started, duration: entry.duration });
      }
    });
    observer.observe({ type: 'longtask', buffered: false });
    window.fetch = async (input, init) => {
      const sync = String(input).endsWith('/api/sim/sync');
      const requestAt = performance.now();
      if (sync) {
        const args = JSON.parse(init.body);
        probe.reports.push(...args.reports.map(report => ({ ...report, seconds: (requestAt-started)/1000 })));
      }
      try {
        const response = await nativeFetch(input, init);
        if (sync) {
          const body = await response.clone().json(), args = JSON.parse(init.body);
          probe.syncs.push({ seconds: (performance.now()-started)/1000, ms: performance.now()-requestAt,
            fps: args.state.fps, brakeIn: args.state.locomotion.brakeIn,
            playing: args.state.playing, generating: args.state.generating,
            phase: args.state.locomotion.phase, quality: args.state.locomotion.quality,
            goal: body.goal?.status, engine: body.behavior?.engine?.phase,
            mapping: args.state.perception?.mapping,
            commands: body.commands.map(command => ({ id: command.id, ...command.arguments })) });
        }
        return response;
      } catch (error) {
        if (sync) probe.errors.push({ seconds: (performance.now()-started)/1000, message: String(error) });
        throw error;
      }
    };
    const response = await nativeFetch('http://127.0.0.1:8100/api/goals/submit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clientId, title: '连续运动与长程稳定性实测', steps: [{ kind: 'social', timeout: 240 }] }),
    });
    if (!response.ok) throw new Error(await response.text());
    probe.goal = await response.json();
    const timer = setInterval(() => {
      probe.samples.push({ seconds: (performance.now()-started)/1000,
        ...document.querySelector('#viewport').dataset });
    }, 200);
    setTimeout(async () => {
      clearInterval(timer); observer.disconnect(); window.fetch = nativeFetch;
      try {
        const state = await nativeFetch('http://127.0.0.1:8100/api/sim/state');
        if (!state.ok) throw new Error('读取结束状态失败');
        const value = await state.json();
        probe.final = { commands: value.commands.filter(command => command.clientId === clientId),
          goal: value.goals.find(goal => goal.clientId === clientId) };
        const cancel = await nativeFetch('http://127.0.0.1:8100/api/goals/cancel', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ clientId }),
        });
        if (!cancel.ok) throw new Error('测试清理失败');
        probe.cleanup = await cancel.json();
      } catch (error) { probe.errors.push({ message: String(error) }); }
      finally {
        probe.powerSaving = document.querySelector('#power-toggle').checked;
        probe.microphone = !document.querySelector('#mic-stop').disabled;
        probe.done = true;
      }
    }, 180000);
    return { clientId, recordingSeconds: 180 };
  });
}

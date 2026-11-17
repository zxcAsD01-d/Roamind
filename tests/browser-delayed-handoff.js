async page => {
  const reports = [], errors = [], samples = [], planning = [];
  let queued = null, delayNext = 0;
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/world', async route => {
    const response = await route.fetch(), world = await response.json();
    // Fixture placement in the real corridor; no geometry reaches autonomy.
    await route.fulfill({ response, json: { ...world, spawn: [-8, .92, 0] } });
  });
  await page.route('**/api/sim/sync', async route => {
    reports.push(...route.request().postDataJSON().reports);
    const response = await route.fetch(), body = await response.json();
    if (queued) { body.commands.push(queued); queued = null; }
    await route.fulfill({ response, json: body });
  });
  await page.route('**/api/plans/visual-step', async route => {
    const delay = delayNext; delayNext = 0;
    planning.push({ delay, at: Date.now() });
    if (delay) await new Promise(resolve => setTimeout(resolve, delay));
    // Aborted requests must not install a late response.
    if (route.request().failure()) return;
    await route.continue();
  });
  const dataset = () => page.locator('#viewport').evaluate(element => ({ ...element.dataset }));
  const waitForReport = async (id, status) => {
    for (let index = 0; index < 220; index++) {
      samples.push({ at: Date.now(), ...await dataset() });
      if (reports.some(report => report.id === id && report.status === status)) return;
      const failure = reports.find(report => report.id === id && report.status === 'failed');
      if (failure && status !== 'failed') throw new Error(JSON.stringify(failure));
      await page.waitForTimeout(100);
    }
    throw new Error('等待命令状态超时：'+id+'/'+status);
  };
  const waitForPrefetch = async () => {
    for (let index = 0; index < 180; index++) {
      samples.push({ at: Date.now(), ...await dataset() });
      const brake = await page.evaluate(async () => {
        const response = await fetch('http://127.0.0.1:8100/api/sim/state');
        if (!response.ok) throw new Error('读取仿真状态失败');
        const state = await response.json(), clientId = document.querySelector('#viewport').dataset.clientId;
        return state.sessions.find(item => item.id === clientId)?.state.locomotion.brakeIn;
      });
      if (brake > 1.15 && brake <= 1.6) return brake;
      await page.waitForTimeout(100);
    }
    throw new Error('未进入行走预取窗口');
  };
  try {
    await page.goto('http://127.0.0.1:5173/');
    await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
    await page.locator('#ambient-idle').uncheck();
    await page.locator('#ego-toggle').evaluate(element => {
      element.checked = false; element.dispatchEvent(new Event('change', { bubbles: true }));
    });
    queued = { id: 'delay-face', kind: 'face', arguments: { heading: 90, duration: 3 } };
    await waitForReport('delay-face', 'completed');
    queued = { id: 'delay-walk', kind: 'visual', arguments: {
      action: 'forward', relativeBearing: 0, viewHeading: 90, distance: 3, duration: 6, continuous: true } };
    await waitForReport('delay-walk', 'playing');
    const firstBrake = await waitForPrefetch();
    const before = await dataset(); delayNext = 450;
    queued = { id: 'delay-successor', kind: 'visual', arguments: {
      action: 'forward', relativeBearing: 0, viewHeading: 90, distance: 3, duration: 6,
      continuous: true, continuationOf: 'delay-walk' } };
    await waitForReport('delay-successor', 'playing');
    await waitForReport('delay-walk', 'completed');
    const after = await dataset();
    if (reports.some(report => report.id === 'delay-successor' && report.status === 'failed'))
      throw new Error('450ms 延迟仍使续接失败');
    if (Number(after.rootX)-Number(before.rootX) < .1) throw new Error('延迟期间没有继续前进');
    const secondBrake = await waitForPrefetch(); delayNext = 2100;
    queued = { id: 'delay-timeout', kind: 'visual', arguments: {
      action: 'forward', relativeBearing: 0, viewHeading: 90, distance: 2, duration: 4,
      continuous: true, continuationOf: 'delay-successor' } };
    await waitForReport('delay-timeout', 'failed');
    await waitForReport('delay-successor', 'completed');
    await page.waitForTimeout(1300);
    const timeout = reports.find(report => report.id === 'delay-timeout' && report.status === 'failed');
    if (timeout.retainedCommandId !== 'delay-successor' || !timeout.detail.includes('请求超时'))
      throw new Error('超时没有恢复旧动作：'+JSON.stringify(timeout));
    for (const id of ['delay-walk', 'delay-successor']) {
      const terminal = reports.filter(report => report.id === id && ['completed','failed','interrupted'].includes(report.status));
      if (terminal.length !== 1 || terminal[0].status !== 'completed') throw new Error('重复或错误终态：'+JSON.stringify(terminal));
    }
    if (reports.some(report => report.id === 'delay-timeout' && report.status === 'playing'))
      throw new Error('超时响应在稍后错误安装');
    if (errors.length) throw new Error(errors.join(';'));
    return { passed: true, firstBrake, secondBrake, before, after, reports, planning, samples, errors,
      final: await dataset() };
  } finally {
    await page.unroute('**/api/world');
    await page.unroute('**/api/sim/sync');
    await page.unroute('**/api/plans/visual-step');
  }
}

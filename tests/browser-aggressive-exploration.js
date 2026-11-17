async page => {
  const observations = [], samples = [];
  const track = request => {
    if (!request.url().endsWith('/api/sim/sync') || request.method() !== 'POST') return;
    const vision = request.postDataJSON()?.vision;
    if (vision) observations.push({ observedAt: vision.observedAt, mapping: vision.stereo.mapping,
      depthAt: vision.stereo.observedAt, signature: vision.signature.length });
  };
  page.on('request', track);
  await page.route('http://127.0.0.1:8200/**', route => route.fulfill({
    status: 503, contentType: 'application/json', headers: { 'Access-Control-Allow-Origin': '*' },
    body: JSON.stringify({ error: '测试：深度与语义服务不可用' }) }));
  try {
    await page.goto('http://127.0.0.1:5173/');
    await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
    await page.locator('#ambient-idle').uncheck();
    const result = await page.evaluate(async () => {
      const { MovementProgress } = await import('/src/movement_progress.ts');
      const watch = new MovementProgress();
      watch.update('walk', true, 0, 0, 0);
      for (let i = 0; i < 179; i++) {
        if (watch.update('walk', true, 1/60, 0, 0)) throw new Error('死锁过早触发');
      }
      if (!watch.update('walk', true, .02, 0, 0)) throw new Error('死锁未触发');
      for (let i = 0; i < 600; i++) {
        if (watch.update('moving', true, 1/60, i*.005, 0)) throw new Error('正常行走误判死锁');
      }
      for (let i = 0; i < 600; i++) {
        if (watch.update('turning', false, 1/60, 0, 0)) throw new Error('转身误判死锁');
      }
      const clientId = document.querySelector('#viewport').dataset.clientId;
      const response = await fetch('http://127.0.0.1:8100/api/goals/submit', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clientId, title: '感知服务离线仍探索', steps: [{ kind: 'social', timeout: 120 }] }) });
      if (!response.ok) throw new Error(await response.text());
      return { clientId, goal: await response.json() };
    });
    for (let i = 0; i < 125; i++) {
      samples.push(await page.locator('#viewport').evaluate(element => ({ ...element.dataset })));
      await page.waitForTimeout(200);
    }
    const state = await page.evaluate(async clientId => {
      const response = await fetch('http://127.0.0.1:8100/api/sim/state');
      if (!response.ok) throw new Error('读取命令失败');
      const state = await response.json();
      const cancel = await fetch('http://127.0.0.1:8100/api/goals/cancel', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ clientId }) });
      if (!cancel.ok) throw new Error('测试清理失败');
      return { commands: state.commands.filter(command => command.clientId === clientId),
        powerSaving: document.querySelector('#power-toggle').checked,
        microphone: !document.querySelector('#mic-stop').disabled };
    }, result.clientId);
    const distance = samples.slice(1).reduce((sum, sample, index) => sum+Math.hypot(
      Number(sample.rootX)-Number(samples[index].rootX), Number(sample.rootZ)-Number(samples[index].rootZ)), 0);
    if (distance < 1) throw new Error('服务离线后仍被固定在原地：'+distance);
    if (observations.length < 5 || observations.some(item => item.mapping)) throw new Error('本地 RGB 回退不正确');
    if (observations.at(-1).observedAt-observations[0].observedAt < 10) throw new Error('视觉帧未持续更新');
    if (state.commands.some(command => command.detail?.startsWith('视觉避障'))) throw new Error('仍有视觉硬停车');
    if (!state.powerSaving || state.microphone) throw new Error('节能或麦克风默认状态改变');
    return { passed: true, distance, observations, ...state };
  } finally {
    page.off('request', track);
    await page.unroute('http://127.0.0.1:8200/**');
  }
}

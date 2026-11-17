async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  const requests = [];
  const listener = request => {
    if (request.url().includes('/api/generate/stream')) requests.push(JSON.parse(request.postData()));
  };
  page.on('request', listener);
  const call = (path, args) => page.evaluate(async ({ path, args }) => {
    const clientId = document.querySelector('#viewport').dataset.clientId;
    const response = await fetch(`http://127.0.0.1:8100${path}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...args, clientId }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(result));
    return result;
  }, { path, args });
  try {
    await page.locator('#reset-house').click();
    await page.waitForTimeout(650);
    const plan = await call('/api/plans/timeline', { start: [-5, .92, 3.2], steps: [
      { action: 'walk', start: 0, end: 8, destination: [-5, .92, 3.8] },
      { action: 'wave', start: 8, end: 11 },
    ] });
    const command = await call('/api/sim/execute', { planId: plan.id });
    const stages = [];
    for (let attempt = 0; attempt < 350; attempt++) {
      const current = await page.evaluate(async id => {
        const response = await fetch(`http://127.0.0.1:8100/api/sim/commands/${id}`);
        const command = await response.json();
        return { status: command.status, detail: command.detail, ...document.querySelector('#viewport').dataset };
      }, command.id);
      const stage = `${current.status}:${current.motionMode}`;
      if (stages.at(-1) !== stage) stages.push(stage);
      if (current.status === 'failed') throw new Error(current.detail);
      if (current.status === 'completed') {
        if (requests.length !== 1) throw new Error(`特殊动作生成次数异常：${requests.length}`);
        if (!stages.includes('playing:preset') || !stages.includes('playing:ardy')) throw new Error(`分段未完整播放：${stages}`);
        if (Number(current.underflows) !== 0) throw new Error('特殊动作缓冲断粮');
        const generatedPlan = await page.evaluate(async id => {
          const response = await fetch(`http://127.0.0.1:8100/api/plans/${id}`);
          if (!response.ok) throw new Error(`生成规划读取失败：${response.status}`);
          return response.json();
        }, requests[0].planId);
        if (generatedPlan.timeline.some(step => step.action !== 'wave')) throw new Error('普通走路进入扩散规划');
        return { stages, generationRequests: requests.length, prefixFrames: requests[0].history.frames, final: current };
      }
      await page.waitForTimeout(200);
    }
    throw new Error(`混合动作未完成：${stages}`);
  } finally {
    page.off('request', listener);
    await page.locator('#reset-house').click();
  }
}

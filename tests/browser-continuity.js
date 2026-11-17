async (page) => {
  const requests = [];
  const errors = [];
  const requestListener = request => {
    if (request.url().endsWith('/api/generate/stream')) requests.push(request.postDataJSON());
  };
  const errorListener = error => errors.push(error.message);
  page.on('request', requestListener);
  page.on('pageerror', errorListener);
  const ensure = (condition, message) => { if (!condition) throw new Error(message); };
  const generate = async prompt => {
    await page.locator('#prompt-select').selectOption(prompt);
    await page.locator('#generate-motion').click();
  };
  const waitForChunk = async () => {
    await page.waitForFunction(() => document.querySelector('#ardy-status').textContent.startsWith('连贯生成中'), null, { timeout: 15000 });
  };
  const waitForComplete = async () => {
    await page.waitForFunction(() => document.querySelector('#ardy-status').textContent.includes('已保存'), null, { timeout: 20000 });
  };
  try {
    await page.locator('#generation-duration').fill('60');
    await generate('A person walks forward.');
    await waitForChunk();
    await page.waitForFunction(() => Number(document.querySelector('#viewport').dataset.motionTime) > .35,
      null, { timeout: 10000 });
    ensure(await page.locator('#generate-motion').isEnabled(), '切换按钮被锁住');
    await page.locator('#generation-duration').fill('8');
    await generate('A person runs.');
    await page.waitForTimeout(100);
    await generate('A person waves their hand.');
    await waitForComplete();
    const wave = requests.at(-1);
    ensure(wave.prompt === 'A person waves their hand.', '提示词切换失败');
    ensure(wave.history.frames > 0 && wave.history.frames <= 160, '没有传入实际历史');
    const chosen = await page.locator('#motion-select').inputValue();
    ensure(chosen.startsWith('ardy_live_'), '最新生成未成为当前动作');
    const selectedPrompt = await page.evaluate(async id => {
      const entries = await fetch('http://127.0.0.1:8100/api/motions').then(response => response.json());
      return entries.find(entry => entry.id === id)?.prompt;
    }, chosen);
    ensure(selectedPrompt === wave.prompt, '过期动作覆盖了最新动作');

    await generate('A person runs.');
    await waitForChunk();
    await page.locator('#play-motion').click();
    const pausedTime = await page.locator('#motion-time').textContent();
    await waitForComplete();
    ensure((await page.locator('#play-motion').textContent()).trim() === '▶', '生成片段使暂停失效');
    ensure(await page.locator('#motion-time').textContent() === pausedTime, '暂停期间播放时间变化');

    await generate('A person walks forward.');
    await waitForChunk();
    await page.locator('#stop-motion').click();
    await page.waitForTimeout(1200);
    ensure(await page.locator('#motion-time').textContent() === '0.0 s', '停止后旧流重新推进播放');
    ensure((await page.locator('#play-motion').textContent()).trim() === '▶', '停止后旧流自动恢复');

    await page.locator('#play-motion').click();
    await page.locator('#motion-select').selectOption('ardy_s_curve');
    await page.waitForFunction(() => document.querySelector('#ardy-status').textContent.includes('帧 ·'), null, { timeout: 10000 });
    ensure((await page.locator('#play-motion').textContent()).trim() === 'Ⅱ', '切换已有片段后未继续播放');
    ensure(errors.length === 0, errors.join('\n'));
    return {
      passed: ['播放中续接', '快速多次切换', '暂停不被覆盖', '停止取消生成', '已有片段连贯切换'],
      requests: requests.map(request => ({ prompt: request.prompt, revision: request.revision, historyFrames: request.history?.frames })),
      errors,
    };
  } finally {
    page.off('request', requestListener);
    page.off('pageerror', errorListener);
  }
}

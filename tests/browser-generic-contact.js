async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#ambient-idle').uncheck();
  await page.locator('.left-panel > .research-controls > summary').click();
  await page.locator('#ego-fov').fill('95'); await page.locator('#ego-fov').blur();
  await page.locator('#ego-pitch').fill('30');
  await page.locator('.runtime-interactions > summary').click();
  await page.locator('#ocr-board').click();
  const boardAt = Date.now()/1000;
  await page.waitForFunction(after => JSON.parse(document.querySelector('#touch-image').dataset.frame || 'null')?.observedAt > after, boardAt);
  // Choose an unlabelled surface, without requesting OCR or pose-based selection.
  const requests = [];
  const listener = request => {
    if (request.url().endsWith('/api/touch/resolve')) requests.push(request.postDataJSON());
  };
  page.on('request', listener);
  try {
    await page.locator('#touch-label').fill('未分类纹理表面');
    const canvas = page.locator('#touch-image');
    await canvas.scrollIntoViewIfNeeded();
    const bounds = await canvas.boundingBox();
    await canvas.hover();
    await page.waitForFunction(() => {
      const frame = JSON.parse(document.querySelector('#touch-image').dataset.frame || 'null');
      return frame && Date.now()/1000-frame.observedAt < .7;
    });
    const resolved = page.waitForResponse(response => response.url().endsWith('/api/touch/resolve'));
    await canvas.click({ position: { x: bounds.width*280/640, y: bounds.height*250/360 } });
    const response = await resolved;
    if (!response.ok()) throw new Error(await response.text());
    await page.waitForFunction(() => { const target = JSON.parse(document.querySelector('#touch-status').dataset.target || 'null'); return target?.targetId.startsWith('point-') && Date.now()/1000-target.observedAt < 1; });
    const target = await page.locator('#touch-status').evaluate(e => JSON.parse(e.dataset.target));
    if (target.personId || target.text || target.referenceKind !== 'visual-surface') throw new Error('选点依赖人体或文字');
    await page.locator('#touch-hand').selectOption('right_hand');
    await page.locator('#goal-kind').selectOption('contact');
    const started = Date.now()/1000;
    await page.locator('#goal-start').click();
    const samples = [];
    for (let n = 0; n < 700; n++) {
      const goal = await page.locator('#goal-status').evaluate(e => JSON.parse(e.dataset.goal || 'null'));
      const report = await page.locator('#contact-live').evaluate(e => JSON.parse(e.dataset.tracking || 'null'));
      if (report && (!samples.length || report.frame !== samples.at(-1).frame)) samples.push(report);
      if (goal?.createdAt >= started && ['completed', 'failed', 'cancelled'].includes(goal.status)) {
        const result = { goal, target, samples, requests, touch: await page.locator('#touch-status').textContent(),
          clientId: await page.locator('#viewport').getAttribute('data-client-id') };
        await page.screenshot({ path: 'output/playwright/visual-targets/generic-contact.png' });
        console.log(JSON.stringify(result));
        if (goal.status !== 'completed' || !samples.some(s => s.reachedSamples >= 2)) throw new Error(JSON.stringify(result));
        return result;
      }
      await page.waitForTimeout(100);
    }
    throw new Error('通用标注接触任务超时');
  } finally { page.off('request', listener); }
}

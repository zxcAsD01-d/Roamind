async page => {
  const errors = [], frames = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.unroute('http://127.0.0.1:8200/**');
  await page.reload();
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('details.research-controls').evaluateAll(items => items.forEach(item => { item.open = true; }));
  await page.locator('#ambient-idle').uncheck();
  await page.locator('#ego-toggle').uncheck();
  await page.locator('#reset-visual-map').click();
  await page.locator('#ego-toggle').check();
  await page.waitForFunction(() => JSON.parse(document.querySelector('#depth-status').dataset.result || 'null')?.surfaceCompletion,
    null, { timeout: 30000 });
  await page.waitForFunction(() => document.querySelector('#ego-preview').dataset.semanticFrameId,
    null, { timeout: 30000 });
  for (let sample = 0; sample < 12; sample++) {
    frames.push(await page.locator('#ego-preview').evaluate(canvas => ({
      displayed: canvas.dataset.frameId, semantic: canvas.dataset.semanticFrameId,
    })));
    await page.waitForTimeout(100);
  }
  if (frames.some(frame => frame.semantic && frame.displayed !== frame.semantic)) {
    throw new Error('语义标记叠到了其他帧');
  }
  const buffers = await page.evaluate(async () => {
    const { VisualWorldMap } = await import('/src/worldmap.ts');
    const canvas = document.createElement('canvas');
    const view = new VisualWorldMap(canvas);
    const point = { positions: [0, 0, 2], colors: [0xaaaaaa], semantics: [1],
      labels: ['unlabeled', 'wall'], objects: [], totalPoints: 1, inferred: [false] };
    view.updateMap({ ...point, revision: 5 });
    const position = view.geometry.getAttribute('position');
    view.updateMap({ ...point, revision: 6, positions: [1, 0, 2] });
    const reused = position === view.geometry.getAttribute('position');
    view.updateMap({ ...point, revision: 4 });
    const stayedLatest = position.getX(0) === 1;
    view.updateMap({ ...point, mapId: 'new-session', revision: 1, positions: [2, 0, 2] });
    const restartedSessionAccepted = position.getX(0) === 2;
    view.resetView();
    view.updateMap({ ...point, revision: 1 });
    const resetAccepted = position.getX(0) === 0 && view.geometry.drawRange.count === 1;
    view.controls.dispose(); view.renderer.dispose();
    return { reused, stayedLatest, restartedSessionAccepted, resetAccepted };
  });
  if (Object.values(buffers).some(value => !value)) throw new Error(JSON.stringify(buffers));
  const result = await page.evaluate(() => ({
    depth: JSON.parse(document.querySelector('#depth-status').dataset.result),
    map: JSON.parse(document.querySelector('#world-status').dataset.state),
    vision: document.querySelector('#vision-status').textContent,
  }));
  const health = await page.evaluate(async () => (await fetch('http://127.0.0.1:8200/health')).json());
  const compactState = !health.latestVision?.semanticFusion?.map;
  if (!compactState) throw new Error('状态轮询重复携带整份点云');
  if (errors.length) throw new Error(JSON.stringify(errors));
  await page.locator('#world-cloud').evaluate(canvas => {
    for (let parent = canvas.parentElement; parent; parent = parent.parentElement) {
      if (parent instanceof HTMLDetailsElement) parent.open = true;
    }
  });
  await page.locator('#world-cloud').screenshot({ path: 'output/playwright/vision-pipeline-cloud.png' });
  await page.screenshot({ path: 'output/playwright/vision-pipeline.png' });
  return { frames, buffers, compactState, ...result, errors };
}

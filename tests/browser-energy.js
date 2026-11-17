async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#autonomy-toggle').uncheck();
  await page.locator('#ego-toggle').check();
  await page.waitForFunction(() => JSON.parse(document.querySelector('#performance-status').dataset.metrics || '{}').perception?.captures > 2);
  const read = () => page.locator('#performance-status').evaluate(element => JSON.parse(element.dataset.metrics));
  const sample = async enabled => {
    await page.locator('#power-toggle').setChecked(enabled);
    await page.waitForTimeout(3000);
    const before = await read();
    await page.waitForTimeout(15000);
    const after = await read(), seconds = (after.sampledAt-before.sampledAt)/1000;
    return { seconds, active: after.active, framesPerSecond: (after.renderedFrames-before.renderedFrames)/seconds,
      simulationStepsPerSecond: (after.simulationSteps-before.simulationSteps)/seconds,
      captureHz: (after.perception.captures-before.perception.captures)/seconds,
      depthHz: (after.perception.depthRequests-before.perception.depthRequests)/seconds,
      semanticHz: (after.perception.semanticRequests-before.perception.semanticRequests)/seconds,
      cadence: after.perception.cadence };
  };
  try {
    const normal = await sample(false), saving = await sample(true);
    if (saving.active || saving.framesPerSecond > 17 || saving.framesPerSecond < 12) throw new Error(`节能闲置帧率异常 ${JSON.stringify(saving)}`);
    if (saving.captureHz > 1.3 || saving.depthHz > 1.3 || saving.semanticHz > .8) throw new Error('闲置感知未降频');
    if (saving.simulationStepsPerSecond < 59) throw new Error('节流拖慢模拟时钟');
    if (saving.framesPerSecond >= normal.framesPerSecond*.65) throw new Error('渲染负载未显著降低');
    await page.reload();
    await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
    if (!await page.locator('#power-toggle').isChecked()) throw new Error('节能设置未保存');
    return {normal, saving, persistence: true};
  } finally { await page.locator('#power-toggle').check(); }
}

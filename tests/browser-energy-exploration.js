async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#power-toggle').check();
  await page.locator('#ego-toggle').check();
  await page.waitForFunction(() => JSON.parse(document.querySelector('#depth-status').dataset.result || '{}').mapping === true);
  await page.locator('#autonomy-reset').click();
  await page.locator('#autonomy-toggle').check();
  await page.waitForTimeout(2000);
  const read = () => page.locator('#performance-status').evaluate(element => JSON.parse(element.dataset.metrics));
  const before = await read();
  await page.waitForTimeout(10000);
  const after = await read(), seconds = (after.sampledAt-before.sampledAt)/1000;
  const result = {clientId: await page.locator('#viewport').getAttribute('data-client-id'),
    active: after.active, powerSaving: after.powerSaving,
    fps: (after.renderedFrames-before.renderedFrames)/seconds,
    simulationHz: (after.simulationSteps-before.simulationSteps)/seconds,
    captureHz: (after.perception.captures-before.perception.captures)/seconds,
    depthHz: (after.perception.depthRequests-before.perception.depthRequests)/seconds,
    semanticHz: (after.perception.semanticRequests-before.perception.semanticRequests)/seconds,
    cadence: after.perception.cadence};
  if (!result.active || !result.powerSaving || result.fps < 27 || result.fps > 33) throw new Error(`活动帧率异常 ${JSON.stringify(result)}`);
  if (result.depthHz < 3 || result.captureHz < 4 || result.cadence.depth !== 200) throw new Error(`活动感知未恢复 ${JSON.stringify(result)}`);
  if (result.simulationHz < 59 || result.simulationHz > 61) throw new Error('活动模拟时间被限帧影响');
  return result;
}

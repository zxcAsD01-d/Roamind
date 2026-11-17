async (page) => {
  await page.waitForFunction(() => document.querySelector('#autonomy-status').textContent.includes('社交条件已达成'));
  await page.locator('#power-toggle').check();
  await page.waitForTimeout(2500);
  const read = () => page.locator('#performance-status').evaluate(element => JSON.parse(element.dataset.metrics));
  const before = await read();
  await page.waitForTimeout(6000);
  const after = await read(), seconds = (after.sampledAt-before.sampledAt)/1000;
  const result = {powerSaving:after.powerSaving, active:after.active, targetFps:after.targetFps,
    fps:(after.renderedFrames-before.renderedFrames)/seconds,
    simulationHz:(after.simulationSteps-before.simulationSteps)/seconds,
    depthHz:(after.perception.depthRequests-before.perception.depthRequests)/seconds,
    status:await page.locator('#autonomy-status').textContent()};
  if (!result.powerSaving || result.active || result.targetFps !== 15) throw new Error(`停步后未降频 ${JSON.stringify(result)}`);
  if (result.fps < 14 || result.fps > 16 || result.depthHz > 1.4) throw new Error(`闲置频率异常 ${JSON.stringify(result)}`);
  return result;
}

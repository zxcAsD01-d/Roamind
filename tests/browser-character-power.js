async (page) => {
  await page.locator('#ambient-idle').uncheck();
  await page.locator('#power-toggle').check();
  await page.waitForTimeout(3000);
  const read=()=>page.locator('#performance-status').evaluate(e=>JSON.parse(e.dataset.metrics));
  const before=await read();await page.waitForTimeout(6500);const after=await read();
  const seconds=(after.sampledAt-before.sampledAt)/1000;
  const result={seconds,fps:(after.renderedFrames-before.renderedFrames)/seconds,
    simulationHz:(after.simulationSteps-before.simulationSteps)/seconds,
    depthHz:(after.perception.depthRequests-before.perception.depthRequests)/seconds,
    active:after.active,targetFps:after.targetFps,powerSaving:after.powerSaving};
  await page.locator('#ambient-idle').check();
  if(!result.powerSaving||result.active||result.targetFps!==15||result.fps<14||result.fps>16||result.depthHz>1.3)
    throw new Error('统一运行台闲置未节能 '+JSON.stringify(result));
  return result;
}

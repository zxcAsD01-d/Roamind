async (page) => {
  const frames = [];
  const record = request => {
    if (request.url() === 'http://127.0.0.1:8200/api/vision') frames.push(request.postDataJSON().frameId);
  };
  await page.locator('#ego-toggle').check();
  await page.waitForFunction(() => !!document.querySelector('#depth-status').dataset.result);
  try {
    await page.locator('#power-toggle').uncheck();
    await page.route('http://127.0.0.1:8200/api/stereo', route => route.abort());
    await page.waitForTimeout(1000);
    const read = () => page.locator('#ego-preview').evaluate(canvas => Number(canvas.dataset.frameId));
    const before = await read();
    page.on('request', record);
    await page.waitForTimeout(4000);
    const after = await read();
    if (after <= before) throw new Error('双目服务中断时 RGB 采集停止');
    if (new Set(frames).size !== frames.length) throw new Error(`同帧重复语义推理 ${frames}`);
    return {semanticFrameIds: frames, duplicateRequests: 0, rgbFramesAdvanced: after-before};
  } finally {
    page.off('request', record);
    await page.unroute('http://127.0.0.1:8200/api/stereo');
    await page.locator('#power-toggle').check();
  }
}

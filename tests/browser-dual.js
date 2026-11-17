async (page) => {
  const errors = [];
  const onError = error => errors.push(error.message);
  page.on('pageerror', onError);
  try {
    await page.locator('#stop-motion').click();
    await page.locator('#generation-duration').fill('24');
    await page.locator('#prompt-select').selectOption('A person walks forward.');
    const started = Date.now();
    await page.locator('#generate-motion').click();
    await page.waitForFunction(() => Number(document.querySelector('#viewport').dataset.motionTime) > .2,
      null, { timeout: 15000 });
    const startMs = Date.now() - started;
    const samples = [];
    for (let i = 0; i < 48; i++) {
      samples.push(await page.locator('#viewport').evaluate(node => ({
        wall: performance.now(), time: Number(node.dataset.motionTime),
        buffer: Number(node.dataset.bufferSeconds), underflows: Number(node.dataset.underflows),
        rootX: Number(node.dataset.rootX), rootZ: Number(node.dataset.rootZ),
      })));
      await page.waitForTimeout(500);
    }
    const underflows = Math.max(...samples.map(sample => sample.underflows));
    if (underflows !== 0) throw new Error(`长播放发生 ${underflows} 次断粮`);
    const playing = samples.filter(sample => sample.time < 23.8);
    const stalled = playing.slice(1).some((sample, i) => sample.time <= playing[i].time);
    if (stalled) throw new Error('播放时间停滞或回退');
    const distances = playing.slice(1).map((sample, i) =>
      Math.hypot(sample.rootX-playing[i].rootX, sample.rootZ-playing[i].rootZ));
    if (distances.some(distance => !Number.isFinite(distance))) throw new Error('模型位置数据无效');
    let stationarySamples = 0;
    for (const distance of distances) {
      stationarySamples = distance < .002 ? stationarySamples+1 : 0;
      if (stationarySamples >= 2) throw new Error('行走时模型连续一秒停在原地');
    }
    if (errors.length) throw new Error(errors.join('\n'));
    return { duration: 24, startMs, underflows, errors,
      lastTime: samples.at(-1).time, minHalfSecondDistance: Math.min(...distances), samples };
  } finally {
    page.off('pageerror', onError);
  }
}

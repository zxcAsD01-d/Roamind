async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  const generated = [];
  const listener = request => { if (request.url().includes('/api/generate/stream')) generated.push(request.url()); };
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
  const waitCommand = async id => {
    for (let attempt = 0; attempt < 225; attempt++) {
      const result = await page.evaluate(async commandId => {
        const response = await fetch(`http://127.0.0.1:8100/api/sim/commands/${commandId}`);
        return response.json();
      }, id);
      if (result.status === 'failed') throw new Error(result.detail);
      if (result.status === 'completed') return page.locator('#viewport').evaluate(element => ({ ...element.dataset }));
      await page.waitForTimeout(200);
    }
    throw new Error(`命令未完成：${id}`);
  };
  try {
    await page.locator('#reset-house').click();
    await page.waitForTimeout(650);
    const face = await call('/api/sim/face', { heading: 90 });
    const facing = await waitCommand(face.id);
    if (Math.abs(Number(facing.heading)-90) > 1) throw new Error(`实际转身未到达指定朝向：${JSON.stringify(facing)}`);
    const move = await call('/api/sim/move', { destination: [-5, .92, 4.1] });
    const moved = await waitCommand(move.id);
    if (Math.hypot(Number(moved.rootX)+5, Number(moved.rootZ)-4.1) > .3) throw new Error('移动目的地误差过大');
    if (generated.length) throw new Error('普通行走／转身激活了 ARDY');
    await page.route('http://127.0.0.1:8200/**', route => route.abort());
    await page.locator('#ego-toggle').check();
    await page.waitForTimeout(350);
    const before = await page.locator('#ego-preview').evaluate(canvas => Number(canvas.dataset.frameId));
    await page.waitForTimeout(750);
    const preview = await page.locator('#ego-preview').evaluate(canvas => {
      const data = canvas.getContext('2d').getImageData(0, 0, 640, 360).data;
      let minimum = 255, maximum = 0;
      for (let index = 0; index < data.length; index += 16) {
        minimum = Math.min(minimum, data[index]); maximum = Math.max(maximum, data[index]);
      }
      return { frame: Number(canvas.dataset.frameId), pixelRange: maximum-minimum };
    });
    if (preview.frame <= before || preview.pixelRange < 8) throw new Error('感知断线后第一人称 RGB 停止或为空');
    await page.unroute('http://127.0.0.1:8200/**');
    await page.waitForFunction(() => !!document.querySelector('#depth-status')?.dataset.result, null, { timeout: 15000 });
    return { facing, moved, ordinaryGenerationRequests: generated.length, offlinePreview: { before, ...preview }, depthRecovered: true };
  } finally {
    page.off('request', listener);
    await page.unroute('http://127.0.0.1:8200/**');
    await page.locator('#ego-toggle').uncheck();
  }
}

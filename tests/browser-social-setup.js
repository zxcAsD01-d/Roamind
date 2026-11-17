async (page) => {
  await page.locator('#autonomy-toggle').uncheck();
  await page.locator('#ego-toggle').uncheck();
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#reset-house').click();
  await page.locator('#touch-partner').click();
  const clientId = await page.locator('#viewport').getAttribute('data-client-id');
  if (!clientId) throw new Error('测试网页会话尚未建立');
  const command = async (path, args) => {
    const result = await page.evaluate(async ({path, args, clientId}) => {
      const response = await fetch(`http://127.0.0.1:8100${path}`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({...args, clientId}),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? `HTTP ${response.status}`);
      return body;
    }, {path, args, clientId});
    for (let attempt = 0; attempt < 225; attempt++) {
      const status = await page.evaluate(async id => {
        const response = await fetch(`http://127.0.0.1:8100/api/sim/commands/${id}`);
        return response.json();
      }, result.id);
      if (['failed', 'interrupted', 'superseded'].includes(status.status)) throw new Error(status.detail || status.status);
      if (status.status === 'completed') return result.id;
      await page.waitForTimeout(200);
    }
    throw new Error(`测试摆位未完成：${result.id}`);
  };
  // Fixture placement only. The autonomous policy receives stereo RGB and detections.
  const move = await command('/api/sim/move', {destination: [-5, .92, 2.35]});
  const face = await command('/api/sim/face', {heading: 0});
  const position = await page.locator('#viewport').evaluate(element => ({...element.dataset}));
  if (Math.hypot(Number(position.rootX)+5, Number(position.rootZ)-2.35) > .35) {
    throw new Error(`测试摆位不正确：${JSON.stringify(position)}`);
  }
  await page.locator('#ego-toggle').check();
  await page.waitForFunction(() => [...document.querySelector('#touch-person').options].some(option => option.value),
    null, {timeout: 15000});
  await page.locator('#autonomy-reset').click();
  await page.locator('#autonomy-toggle').check();
  return {clientId, setupCommands: [move, face], scenario: 'visible VRM person at conversational distance'};
}

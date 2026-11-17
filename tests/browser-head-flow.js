async page => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#ambient-idle').uncheck();
  await page.waitForFunction(() => JSON.parse(document.querySelector('#depth-status').dataset.result || '{}').mapping);
  const clientId = await page.locator('#viewport').getAttribute('data-client-id');
  const commands = [], poses = [];
  for (const relativeBearing of [-10, 10, -35, 35, 0]) {
    const command = await page.evaluate(async ({clientId, relativeBearing}) => {
      const response = await fetch('http://127.0.0.1:8100/api/sim/look', {method:'POST',
        headers:{'Content-Type':'application/json'}, body:JSON.stringify({clientId, relativeBearing, duration:.4})});
      const result = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(result));
      return result;
    }, {clientId, relativeBearing});
    const samples = [];
    for (let index = 0; index < 100; index++) {
      const sample = await page.evaluate(async id => {
        const response = await fetch('http://127.0.0.1:8100/api/sim/commands/'+id);
        if (!response.ok) throw new Error('command HTTP '+response.status);
        return {command:await response.json(), root:{...document.querySelector('#viewport').dataset},
          depth:JSON.parse(document.querySelector('#depth-status').dataset.result || '{}')};
      }, command.id);
      samples.push(sample);
      if (sample.command.status === 'completed') break;
      if (['failed','superseded','interrupted'].includes(sample.command.status)) throw new Error(sample.command.detail);
      await page.waitForTimeout(80);
    }
    const last = samples.at(-1);
    if (last.command.status !== 'completed') throw new Error('转头超时');
    if (Math.abs(Number(last.root.headYaw)+relativeBearing) > 2) throw new Error('转头方向或幅度错误 '+JSON.stringify(last.root));
    if (samples.some(item => Math.hypot(Number(item.root.rootX)-Number(last.root.rootX),Number(item.root.rootZ)-Number(last.root.rootZ)) > .002))
      throw new Error('转头带动了根节点');
    if (samples.some(item => Math.abs(Number(item.root.headSpeed)) > 65.01)) throw new Error('转头超出限速');
    await page.waitForTimeout(350);
    poses.push(await page.evaluate(() => ({root:{...document.querySelector('#viewport').dataset},
      depth:JSON.parse(document.querySelector('#depth-status').dataset.result || '{}'),
      pixels:Array.from(document.querySelector('#ego-preview').getContext('2d').getImageData(0,0,640,360).data)
        .filter((_,index)=>index%64<3)})));
    commands.push({relativeBearing, seconds:samples.length*.08, samples:samples.length, end:last.root});
  }
  const difference = (a,b) => a.reduce((sum,value,index) => sum+Math.abs(value-b[index]),0)/a.length;
  const pixelDifferences = [difference(poses[0].pixels,poses[1].pixels),difference(poses[2].pixels,poses[3].pixels)];
  if (pixelDifferences.some(value => value < 2)) throw new Error('左右转头未改变真实第一人称画面');
  return {passed:true,commands,pixelDifferences,microphoneClosed:await page.locator('#mic-stop').isDisabled()};
}

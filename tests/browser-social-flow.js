async page => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('details').evaluateAll(items => items.forEach(item => { item.open = true; }));
  await page.locator('#ambient-idle').uncheck();
  await page.locator('#ego-toggle').uncheck();
  await page.locator('#touch-partner').click();
  const clientId = await page.locator('#viewport').getAttribute('data-client-id');
  const post = (path, args = {}) => page.evaluate(async ({path, args, clientId}) => {
    const response = await fetch('http://127.0.0.1:8100'+path, {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({...args, clientId})});
    const value = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(value));
    return value;
  }, {path, args, clientId});
  const finish = async command => {
    for (let index = 0; index < 160; index++) {
      const state = await page.evaluate(async id => (await fetch('http://127.0.0.1:8100/api/sim/commands/'+id)).json(), command.id);
      if (state.status === 'completed') return;
      if (['failed','interrupted','superseded'].includes(state.status)) throw new Error(state.detail);
      await page.waitForTimeout(250);
    }
    throw new Error('社交测试摆位超时');
  };
  // Fixture placement only; no partner coordinates enter the visual policy.
  await finish(await post('/api/sim/move', {destination:[-5,.92,1.7]}));
  await finish(await post('/api/sim/face', {heading:0}));
  await page.locator('#reset-visual-map').click();
  await page.locator('#ego-toggle').check();
  await page.waitForFunction(() => JSON.parse(document.querySelector('#depth-status').dataset.result || '{}').mapping);
  await page.evaluate(() => { window.socialFlowProbe = {samples:[], commands:{}, errors:[], done:false}; });
  await post('/api/goals/submit', {title:'真实双目人物接近与挥手实测',steps:[{kind:'social',timeout:90}]});
  return page.evaluate(clientId => {
    const probe = window.socialFlowProbe, started = performance.now();
    let polling = false;
    const timer = setInterval(async () => {
      if (polling) return;
      polling = true;
      try {
        const postState = await fetch('http://127.0.0.1:8100/api/goals/state', {method:'POST',
          headers:{'Content-Type':'application/json'}, body:JSON.stringify({clientId})});
        if (!postState.ok) throw new Error('goal state HTTP '+postState.status);
        const state = await postState.json();
        const response = await fetch('http://127.0.0.1:8100/api/sim/state');
        if (!response.ok) throw new Error('sim state HTTP '+response.status);
        const sim = await response.json();
        probe.samples.push({seconds:(performance.now()-started)/1000,state,root:{...document.querySelector('#viewport').dataset},
          partner:JSON.parse(document.querySelector('#touch-partner').dataset.position || 'null')});
        sim.commands.filter(c => c.clientId === clientId).forEach(c => {probe.commands[c.id]=c;});
        if (['completed','failed','cancelled'].includes(state.goal.status)) {clearInterval(timer);probe.final=state;probe.done=true;}
      } catch(error) { probe.errors.push(String(error)); }
      finally { polling = false; }
    },500);
    setTimeout(async () => {
      clearInterval(timer);
      if (probe.done) return;
      try {
        const response = await fetch('http://127.0.0.1:8100/api/goals/cancel',{method:'POST',
          headers:{'Content-Type':'application/json'},body:JSON.stringify({clientId})});
        if (!response.ok) throw new Error('cancel HTTP '+response.status);
        probe.cleanup=await response.json();
      } catch(error) { probe.errors.push(String(error)); }
      finally { probe.done=true; }
    },60000);
    return {clientId,recordingSeconds:60};
  },clientId);
}

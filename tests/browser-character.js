async (page) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#viewport')?.dataset.clientId);
  await page.locator('#power-toggle').check();
  const clientId = await page.locator('#viewport').getAttribute('data-client-id');
  const request = (path, args) => page.evaluate(async ({path, args, clientId}) => {
    const response = await fetch('http://127.0.0.1:8100'+path, {method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({...args,clientId})});
    const body=await response.json(); if(!response.ok)throw new Error(JSON.stringify(body)); return body;
  }, {path,args,clientId});
  const goal = () => page.locator('#goal-status').evaluate(element => JSON.parse(element.dataset.goal || 'null'));
  const wait = async (id, allowed=['completed']) => {
    for(let index=0;index<350;index++) {
      const result=await goal();
      if(result?.id===id && ['completed','failed','cancelled'].includes(result.status)) {
        if(!allowed.includes(result.status)) throw new Error('任务未通过 '+JSON.stringify(result));
        return result;
      }
      await page.waitForTimeout(100);
    }
    throw new Error('任务超时 '+JSON.stringify(await goal()));
  };
  await page.locator('#ambient-idle').uncheck();
  await page.locator('.runtime-composer summary').click();
  await page.locator('#goal-json').fill(JSON.stringify({title:'连贯动作验收',steps:[
    {kind:'wait',duration:1.2},{kind:'face',heading:30},{kind:'wait',duration:1.2}]}));
  await page.locator('#goal-json-run').click();
  await page.waitForFunction(()=>JSON.parse(document.querySelector('#goal-status').dataset.goal||'null')?.title==='连贯动作验收');
  const start=await goal(), samples=[];
  for(let index=0;index<250;index++) {
    const state=await page.locator('#viewport').evaluate(e=>({...e.dataset}));
    samples.push({time:Date.now(),x:Number(state.rootX),z:Number(state.rootZ),phase:state.phase,mode:state.motionMode});
    const current=await goal();
    if(current?.id===start.id && ['completed','failed'].includes(current.status))break;
    await page.waitForTimeout(100);
  }
  const sequence=await wait(start.id);
  if(sequence.steps.some(step=>step.status!=='completed'))throw new Error('组合步骤未全部完成');
  await page.waitForFunction(()=>document.querySelector('#character-memory').textContent.includes('新鲜观测'),null,{timeout:15000});
  const missing=await request('/api/goals/submit',{title:'未知位置拒绝',steps:[{kind:'waypoint',position:[999,0,999]}]});
  const refused=await wait(missing.id,['failed']);
  if(!refused.detail.includes('视觉可通行路径'))throw new Error('未知路径未明确拒绝');
  const social=await request('/api/goals/submit',{title:'探索取消验收',steps:[{kind:'social',timeout:45},{kind:'action',action:'wave'}]});
  await page.waitForTimeout(6500);
  const exploration=await request('/api/goals/state',{});
  await page.locator('#goal-cancel').click();
  const cancelled=await wait(social.id,['cancelled']);
  await page.waitForTimeout(3000);
  const later=await request('/api/goals/state',{});
  if(later.goal.status!=='cancelled'||later.goal.steps[1].status!=='pending')throw new Error('取消后仍执行后续动作');
  await page.locator('#ambient-idle').check();
  const idlePhases=[];
  for(let index=0;index<150;index++) {
    idlePhases.push(await page.locator('#viewport').evaluate(e=>({phase:e.dataset.phase,time:e.dataset.motionTime,mode:e.dataset.motionMode})));
    await page.waitForTimeout(100);
  }
  const movingIdle=idlePhases.some((item,index)=>index>0&&item.phase==='idle'&&item.time!==idlePhases[index-1].time);
  if(!movingIdle)throw new Error('自然待机未播放');
  if(errors.length)throw new Error('浏览器异常 '+errors.join('; '));
  await page.screenshot({path:'output/playwright/character/runtime.png'});
  return {clientId,sequence,refused,cancelled,exploration:exploration.behavior.spatialMemory,
    afterCancel:later.goal, samples,idlePhases,errors,
    performance:await page.locator('#performance-status').evaluate(e=>JSON.parse(e.dataset.metrics))};
}

async page => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'), null, { timeout: 60000 });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const request = async (path, body={}) => page.evaluate(async ({path,body}) => {
    const clientId = document.querySelector('#viewport').dataset.clientId;
    const response = await fetch('http://127.0.0.1:8100'+path, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({clientId,...body}) });
    const result = await response.json(); if (!response.ok) throw new Error(JSON.stringify(result)); return result;
  }, {path,body});
  await page.waitForFunction(() => document.querySelector('#viewport')?.dataset.clientId);
  const geometry = await page.evaluate(async () => {
    const THREE = await import('/node_modules/.vite/deps/three.js');
    const {HouseScene} = await import('/src/house.ts');
    const world = await (await fetch('http://127.0.0.1:8100/api/world')).json();
    const house = new HouseScene(world,new THREE.Scene());
    let sprites=0, batches=0, shadowReceivers=0;
    house.group.traverse(object => { sprites+=Number(object.isSprite===true);batches+=Number(object.isInstancedMesh===true);shadowReceivers+=Number(object.receiveShadow); });
    if(sprites!==0)throw new Error('仍存在漂浮标签');
    if(world.objects.filter(o=>o.sourceId==='dining-table').length!==5)throw new Error('桌面/桌腿几何未生效');
    return {sprites,batches,shadowReceivers,solids:world.objects.length};
  });
  const intent = {requestId:'browser-wave-'+Date.now(),intent:{kind:'action',action:'wave',duration:3}};
  const first = await request('/api/agent/intent',intent), repeat = await request('/api/agent/intent',intent);
  if(!repeat.replayed||repeat.goal.id!==first.goal.id)throw new Error('高层目标重试不是幂等的');
  let completed;
  for(let i=0;i<80;i++){
    completed=await request('/api/agent/context');
    if(['completed','failed','cancelled'].includes(completed.goal?.status))break;
    await page.waitForTimeout(200);
  }
  if(completed.goal?.status!=='completed')throw new Error('实际动作未完成 '+JSON.stringify(completed.goal));
  if('rooms' in completed.world||'bounds' in completed.world)throw new Error('高层感知泄漏场景真值');
  await page.getByRole('checkbox',{name:'自主生活 · 探索与社交'}).check();
  const samples=[];
  for(let i=0;i<30;i++){
    await page.waitForTimeout(1500);
    const context=await request('/api/agent/context');
    const state=await page.evaluate(async()=>{
      const id=document.querySelector('#viewport').dataset.clientId;
      const d=await(await fetch('http://127.0.0.1:8100/api/sim/state')).json();
      return d.sessions.find(s=>s.id===id)?.state;
    });
    samples.push({at:Date.now(),kernel:context.kernel,goal:context.goal,
      visual:context.world.self,position:state.position,fps:state.fps,
      objects:context.interactions.objects.length,execution:context.execution});
  }
  await page.getByRole('checkbox',{name:'自主生活 · 探索与社交'}).uncheck();
  const paused=await request('/api/agent/context');
  if(paused.kernel.enabled)throw new Error('自主模式未关闭');
  const origin=samples[0].position;
  const distance=Math.max(...samples.map(s=>Math.hypot(s.position[0]-origin[0],s.position[2]-origin[2])));
  const report={geometry,gesture:{id:first.goal.id,status:completed.goal.status,replayed:repeat.replayed},samples,maxDisplacement:distance,paused:paused.kernel,errors};
  await page.screenshot({path:'output/environment-upgrade/after-autonomy.png'});
  if(errors.length)throw new Error(errors.join(';'));
  return report;
}

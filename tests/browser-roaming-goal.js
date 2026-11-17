async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(()=>document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.waitForFunction(()=>JSON.parse(document.querySelector('#depth-status').dataset.result||'{}').mapping===true);
  await page.locator('#ambient-idle').uncheck();
  const clientId=await page.locator('#viewport').getAttribute('data-client-id');
  const request=(path,args)=>page.evaluate(async ({path,args,clientId})=>{
    const response=await fetch('http://127.0.0.1:8100'+path,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({...args,clientId})});
    const value=await response.json();if(!response.ok)throw new Error(JSON.stringify(value));return value;
  },{path,args,clientId});
  const before=await request('/api/goals/state',{});
  const task=await request('/api/goals/submit',{title:'长距离视觉探索验收',steps:[{kind:'social',timeout:120}]});
  const samples=[], commands=new Map(), observations=[];
  for(let index=0;index<300;index++) {
    samples.push(await page.locator('#viewport').evaluate(e=>({time:Date.now(),...e.dataset})));
    if(index%4===0) {
      const observed=await request('/api/goals/state',{});
      const sim=await page.evaluate(async()=>(await fetch('http://127.0.0.1:8100/api/sim/state')).json());
      sim.commands.filter(c=>c.clientId===clientId).forEach(c=>commands.set(c.id,c));
      observations.push({seconds:index/4,goal:observed.goal,mission:observed.behavior.mission,
        memory:observed.behavior.spatialMemory,depth:await page.locator('#depth-status').evaluate(e=>JSON.parse(e.dataset.result))});
    }
    await page.waitForTimeout(250);
  }
  const after=await request('/api/goals/state',{});
  await request('/api/goals/cancel',{});
  await page.waitForTimeout(2500);
  const final=await request('/api/goals/state',{});
  let distance=0,maxStep=0;
  for(let i=1;i<samples.length;i++) {
    const step=Math.hypot(Number(samples[i].rootX)-Number(samples[i-1].rootX),Number(samples[i].rootZ)-Number(samples[i-1].rootZ));
    distance+=step;maxStep=Math.max(maxStep,step);
  }
  if(final.goal.status!=='cancelled'&&final.goal.status!=='failed')throw new Error('游走结束后仍未停');
  if(samples.some(s=>Number(s.underflows)>0))throw new Error('预设游走发生动作缓冲耗尽');
  const origin=samples[0], displacement=s=>Math.hypot(Number(s.rootX)-Number(origin.rootX),Number(s.rootZ)-Number(origin.rootZ));
  const maxDisplacement=Math.max(...samples.map(displacement)),netDisplacement=displacement(samples.at(-1));
  const forwards=[...commands.values()].filter(c=>c.kind==='visual'&&c.arguments.action==='forward');
  const summary={netDisplacement,maxDisplacement,forwardCommands:forwards.length,
    completedForward:forwards.filter(c=>c.status==='completed').length,
    trackingRatio:observations.filter(o=>o.depth.mapping).length/observations.length,
    regions:new Set(samples.map(s=>`${Math.floor(Number(s.rootX)/.7)},${Math.floor(Number(s.rootZ)/.7)}`)).size};
  if(after.goal.status==='failed'||maxDisplacement<2.5||summary.forwardCommands<5
    ||summary.trackingRatio<.9||summary.regions<8)throw new Error('持续探索未通过 '+JSON.stringify(summary));
  return {summary,commands:[...commands.values()],observations,taskId:task.id,before:before.behavior.spatialMemory,after:after.behavior.spatialMemory,
    mission:after.behavior.mission,status:after.goal,final:final.goal,distance,maxStep,samples,
    performance:await page.locator('#performance-status').evaluate(e=>JSON.parse(e.dataset.metrics))};
}

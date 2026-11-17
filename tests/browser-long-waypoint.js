async page => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(()=>document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#ambient-idle').uncheck();
  await page.waitForFunction(()=>JSON.parse(document.querySelector('#depth-status').dataset.result||'{}').mapping===true);
  await page.waitForFunction(()=>document.querySelector('#character-memory').textContent.includes('新鲜观测'));
  const clientId=await page.locator('#viewport').getAttribute('data-client-id');
  const request=(path,args={})=>page.evaluate(async({path,args,clientId})=>{
    const response=await fetch('http://127.0.0.1:8100'+path,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({...args,clientId})});
    const value=await response.json();if(!response.ok)throw new Error(JSON.stringify(value));return value;
  },{path,args,clientId});
  const memory=(await request('/api/goals/state')).behavior.spatialMemory;
  const pose=await page.locator('#depth-status').evaluate(e=>JSON.parse(e.dataset.result).pose.cameraToWorld);
  const r=memory.resolution,position=memory.position,key=p=>p.join(',');
  const neighbors=([x,z])=>[[x-1,z],[x+1,z],[x,z-1],[x,z+1]];
  const free=new Set(memory.knownCells.filter(c=>c[2]<0).map(c=>key(c.slice(0,2))));
  const blocked=new Set(memory.knownCells.filter(c=>c[2]>0).map(c=>key(c.slice(0,2))));
  const queue=[[Math.floor(position[0]/r),Math.floor(position[2]/r)]],seen=new Set(queue.map(key)),candidates=[];
  for(let index=0;index<queue.length&&index<6000;index++)for(const next of neighbors(queue[index])) {
    if(seen.has(key(next))||!free.has(key(next))||[next,...neighbors(next)].some(p=>blocked.has(key(p))))continue;
    seen.add(key(next));queue.push(next);
    const x=(next[0]+.5)*r,z=(next[1]+.5)*r,dx=x-position[0],dz=z-position[2];
    const distance=Math.hypot(dx,dz),angle=Math.abs(Math.atan2(dx*pose[10]-dz*pose[2],dx*pose[2]+dz*pose[10]));
    if(distance>=1.8&&distance<=2.6&&angle<.18)candidates.push({position:[x,0,z],distance});
  }
  candidates.sort((a,b)=>b.distance-a.distance);
  if(!candidates.length)throw new Error('没有足够远的视觉已知目标 '+JSON.stringify(memory));
  const target=candidates[0],samples=[],commands=new Map();
  await page.evaluate(() => { window.longWaypointProbe = { samples: [], commands: [] }; });
  await request('/api/goals/submit',{title:'视觉长距离目标验收',steps:[{kind:'waypoint',position:target.position,timeout:90}]});
  let final;
  try {
    for(let index=0;index<360;index++) {
      const state=await request('/api/goals/state');
      samples.push({time:Date.now(),root:await page.locator('#viewport').evaluate(e=>({...e.dataset})),
        position:state.behavior.spatialMemory.position,status:state.goal.status,detail:state.goal.detail});
      const sim=await page.evaluate(async()=>(await fetch('http://127.0.0.1:8100/api/sim/state')).json());
      sim.commands.filter(c=>c.clientId===clientId).forEach(c=>commands.set(c.id,c));
      final=state;
      if(['completed','failed','cancelled'].includes(state.goal.status))break;
      await page.waitForTimeout(250);
    }
  } finally {if(final?.goal.status!=='completed')await request('/api/goals/cancel');}
  const a=samples[0].root,b=samples.at(-1).root;
  const netMeters=Math.hypot(Number(a.rootX)-Number(b.rootX),Number(a.rootZ)-Number(b.rootZ));
  const forward=[...commands.values()].filter(c=>c.arguments.action==='forward');
  const summary={targetMeters:target.distance,netMeters,status:final.goal.status,
    forwardCommands:forward.length,completedForward:forward.filter(c=>c.status==='completed').length,
    seconds:(samples.at(-1).time-samples[0].time)/1000};
  await page.evaluate(probe => { window.longWaypointProbe = probe; }, {summary, final, samples, commands:[...commands.values()]});
  if(summary.status!=='completed'||netMeters<1.5||summary.forwardCommands!==1||summary.completedForward!==1)
    throw new Error('长距离视觉目标未通过 '+JSON.stringify({summary,detail:final.goal.detail}));
  return {passed:true,summary,target,final,samples,commands:[...commands.values()]};
}

async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(()=>document.querySelector('#viewport')?.dataset.clientId);
  await page.locator('#ambient-idle').uncheck();
  const clientId=await page.locator('#viewport').getAttribute('data-client-id');
  const request=(path,args)=>page.evaluate(async ({path,args,clientId})=>{
    const response=await fetch('http://127.0.0.1:8100'+path,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({...args,clientId})});
    const value=await response.json();if(!response.ok)throw new Error(JSON.stringify(value));return value;
  },{path,args,clientId});
  const waitCommand=async command=>{
    for(let n=0;n<250;n++) {
      const result=await page.evaluate(async id=>(await fetch('http://127.0.0.1:8100/api/sim/commands/'+id)).json(),command.id);
      if(result.status==='completed')return result;
      if(['failed','superseded','interrupted'].includes(result.status))throw new Error(JSON.stringify(result));
      await page.waitForTimeout(100);
    }
    throw new Error('摆位超时');
  };
  await page.waitForFunction(()=>document.querySelector('#character-memory').textContent.includes('新鲜观测'),null,{timeout:15000});
  const memory=(await request('/api/goals/state',{})).behavior.spatialMemory;
  const pose=await page.locator('#depth-status').evaluate(e=>JSON.parse(e.dataset.result).pose.cameraToWorld);
  const r=memory.resolution,position=memory.position,start=[Math.floor(position[0]/r),Math.floor(position[2]/r)];
  const key=p=>p.join(','), free=new Set(memory.knownCells.filter(c=>c[2]<0).map(c=>key(c.slice(0,2))));
  const blocked=new Set(memory.knownCells.filter(c=>c[2]>0).map(c=>key(c.slice(0,2))));
  const neighbors=([x,z])=>[[x-1,z],[x+1,z],[x,z-1],[x,z+1]];
  const seen=new Set([key(start)]),queue=[start],candidates=[];
  for(let i=0;i<queue.length&&i<6000;i++)for(const next of neighbors(queue[i])) {
    if(seen.has(key(next))||!free.has(key(next))||[next,...neighbors(next)].some(p=>blocked.has(key(p))))continue;
    seen.add(key(next));queue.push(next);
    const x=(next[0]+.5)*r,z=(next[1]+.5)*r,dx=x-position[0],dz=z-position[2];
    const distance=Math.hypot(dx,dz),angle=Math.abs(Math.atan2(dx*pose[10]-dz*pose[2],dx*pose[2]+dz*pose[10]));
    if(distance>=.5&&distance<=1&&angle<.22)candidates.push({position:[x,0,z],distance,angle});
  }
  candidates.sort((a,b)=>b.distance-a.distance);
  if(!candidates.length)throw new Error('当前视野没有前方安全已知目标 '+JSON.stringify(memory));
  const target=candidates[0],task=await request('/api/goals/submit',{title:'视觉已知位置移动验收',steps:[{kind:'waypoint',position:target.position,timeout:35}]});
  const samples=[];let arrived=null;
  for(let n=0;n<350;n++) {
    const result=await request('/api/goals/state',{});
    samples.push({status:result.goal.status,detail:result.goal.detail,position:result.behavior.spatialMemory.position});
    if(['completed','failed','cancelled'].includes(result.goal.status)){arrived=result;break;}
    await page.waitForTimeout(100);
  }
  if(arrived?.goal.status!=='completed')throw new Error('视觉到达未通过 '+JSON.stringify({target,arrived,samples:samples.slice(-10)}));
  await waitCommand(await request('/api/sim/interrupt',{mode:'reset'}));
  await page.locator('.runtime-interactions summary').click();
  await page.locator('#touch-partner').click();
  await page.waitForFunction(()=>document.querySelector('#touch-status').textContent.includes('已添加'));
  await waitCommand(await request('/api/sim/move',{destination:[-5,.92,2.35]}));
  await waitCommand(await request('/api/sim/face',{heading:0}));
  const social=await request('/api/goals/submit',{title:'找人条件与后续动作验收',steps:[{kind:'social',timeout:35},{kind:'wait',duration:1}]});
  let socialResult=null;
  for(let n=0;n<400;n++) {
    const result=await request('/api/goals/state',{});
    if(['completed','failed','cancelled'].includes(result.goal.status)){socialResult=result;break;}
    await page.waitForTimeout(100);
  }
  if(socialResult?.goal.status!=='completed')throw new Error('社交任务未完成 '+JSON.stringify(socialResult));
  return {task:task.id,target,arrived,samples,social:social.id,socialResult};
}

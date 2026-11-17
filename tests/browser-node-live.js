async page => {
  await page.route('**/api/world', async route => {
    const response=await route.fetch(); const world=await response.json();
    world.spawn=[-7.7,.92,4.7];
    await route.fulfill({response,json:world});
  });
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(()=>document.querySelector('#load-status')?.textContent.includes('步幅比例'),null,{timeout:60000});
  await page.locator('#ambient-idle').uncheck();
  const request=async(path,body={})=>page.evaluate(async({path,body})=>{
    const clientId=document.querySelector('#viewport').dataset.clientId;
    const r=await fetch('http://127.0.0.1:8100'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clientId,...body})});
    const result=await r.json();if(!r.ok)throw new Error(JSON.stringify(result));return result;
  },{path,body});
  const wait=async(id)=>{for(let i=0;i<180;i++){
    const r=await page.evaluate(async(id)=>await(await fetch('http://127.0.0.1:8100/api/sim/commands/'+id)).json(),id);
    if(['completed','failed','superseded','interrupted'].includes(r.status))return r;
    await page.waitForTimeout(200);
  }throw new Error('command timeout');};
  await page.waitForTimeout(2000);
  const moved={status:'fixture-placement',note:'测试摆位，不属于视觉规划结果'};
  const face=await request('/api/sim/face',{heading:-90,duration:2});
  const faced=await wait(face.id);
  const look=await request('/api/sim/look',{relativeBearing:0,viewPitch:-25,duration:.5});
  const looked=await wait(look.id);
  await page.locator('#ego-toggle').check();
  const samples=[];let chosen=null;
  for(let i=0;i<24;i++){
    await page.waitForTimeout(1000);
    const ctx=await request('/api/agent/context');
    samples.push({world:ctx.world.self,objects:ctx.interactions.objects,needs:ctx.kernel.needs});
    chosen=ctx.interactions.objects.find(row=>row.options.some(option=>option.operation==='sit'&&option.enabled));
    if(chosen)break;
  }
  let goal=null,result=null;
  if(chosen){
    goal=await request('/api/agent/intent',{requestId:'live-seat-'+Date.now(),intent:{kind:'interact',objectId:chosen.objectId,operation:'sit'}});
    for(let i=0;i<100;i++){
      await page.waitForTimeout(250);result=await request('/api/agent/context');
      if(['completed','failed','cancelled'].includes(result.goal?.status))break;
    }
  }
  await page.screenshot({path:'output/needs-interaction/live-scene.png'});
  await page.unroute('**/api/world');
  return {passed:result?.goal?.status==='completed',moved,faced,looked,samples,chosen:chosen??null,goal,result};
}

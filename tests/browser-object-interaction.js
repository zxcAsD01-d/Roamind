async page => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const request = (path, args={}) => page.evaluate(async ({path,args}) => {
    const clientId=document.querySelector('#viewport').dataset.clientId;
    const response=await fetch('http://127.0.0.1:8100'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clientId,...args})});
    const value=await response.json();if(!response.ok)throw new Error(JSON.stringify(value));return value;
  }, {path,args});
  let catalog;
  for(let i=0;i<40;i++){
    catalog=await request('/api/interactions/options');
    if(catalog.objects.some(row=>row.fresh&&row.options.some(option=>option.operation==='inspect'&&option.enabled)))break;
    await page.waitForTimeout(300);
  }
  const row=[...catalog.objects].sort((a,b)=>(b.label==='person')-(a.label==='person')).find(row=>row.fresh&&row.options.some(option=>option.operation==='inspect'&&option.enabled));
  if(!row)throw new Error('真实视觉未产生可复核物体 '+JSON.stringify(catalog));
  if(await page.locator('#interaction-object option').count()===0)throw new Error('选项未出现在主页');
  const plan=await request('/api/interactions/plan',{objectId:row.objectId,operation:'inspect'});
  if(plan.steps[0].kind!=='interact'||!plan.revalidateAtExecution)throw new Error('规划协议错误');
  await page.locator('#interaction-object').selectOption(row.objectId);
  await page.locator('#interaction-options button').filter({hasText:/^观察$/}).click();
  await page.waitForFunction(()=>JSON.parse(document.querySelector('#goal-status').dataset.goal||'null')?.steps[0].kind==='interact');
  const task=(await request('/api/goals/state')).goal;
  let final;
  for(let i=0;i<120;i++){
    final=(await request('/api/goals/state')).goal;
    if(['completed','failed','cancelled'].includes(final.status))break;
    await page.waitForTimeout(150);
  }
  if(final?.id!==task.id||final.status!=='completed')throw new Error('物体观察未完成 '+JSON.stringify(final));
  if(!final.events.some(event=>event.status==='observe'))throw new Error('跳过了真实观察阶段');
  if(errors.length)throw new Error(errors.join(';'));
  await page.screenshot({path:'output/lumen-rebuild/object-interaction.png'});
  return {passed:true,catalog,plan,final,errors};
}

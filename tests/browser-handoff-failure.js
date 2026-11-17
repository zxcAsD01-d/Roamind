async page => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#ambient-idle').uncheck();
  await page.locator('#ego-toggle').evaluate(n => { n.checked=false; n.dispatchEvent(new Event('change',{bubbles:true})); });
  const reports=[], errors=[]; let queued=null, rejectNext=false;
  page.on('pageerror',error=>errors.push(error.message));
  await page.route('**/api/sim/sync',async route=>{
    reports.push(...JSON.parse(route.request().postData()).reports);
    const response=await route.fetch(), body=await response.json();
    if(queued){body.commands.push(queued);queued=null;}
    await route.fulfill({response,json:body});
  });
  await page.route('**/api/plans/visual-step',async route=>{
    if(!rejectNext){await route.continue();return;}
    rejectNext=false;
    await route.fulfill({status:422,contentType:'application/json',body:JSON.stringify({error:'fault-injection: rejected continuation'})});
  });
  const dataset=()=>page.locator('#viewport').evaluate(n=>({...n.dataset}));
  try {
    const initial=await dataset(), heading=Number(initial.heading);
    queued={id:'test-retained-turn',kind:'visual',arguments:{action:'turn_left',relativeBearing:-65,viewHeading:heading,duration:2}};
    await page.waitForFunction(()=>document.querySelector('#viewport').dataset.phase==='turn');
    await page.waitForTimeout(350);
    const before=await dataset(); rejectNext=true;
    queued={id:'test-rejected-turn',kind:'visual',arguments:{action:'forward',relativeBearing:0,viewHeading:heading+65,
      distance:2,duration:4,continuationOf:'test-retained-turn'}};
    for(let i=0;i<100&&!reports.some(r=>r.id==='test-retained-turn'&&r.status==='completed');i++)await page.waitForTimeout(100);
    const after=await dataset();
    const failed=reports.find(r=>r.id==='test-rejected-turn'&&r.status==='failed');
    if(failed?.retainedCommandId!=='test-retained-turn')throw new Error('失败未保留旧动作 '+JSON.stringify(reports));
    if(Number(after.motionTime)<=Number(before.motionTime)+.1)throw new Error('失败后播放时间停止');
    if(Math.abs(Number(after.heading)-heading-65)>2)throw new Error('旧转身未完整播放 '+JSON.stringify(after));
    const terminal=reports.filter(r=>r.id==='test-retained-turn'&&['completed','failed','interrupted'].includes(r.status));
    if(terminal.length!==1||terminal[0].status!=='completed')throw new Error('旧动作重复或错误终态 '+JSON.stringify(terminal));
    if(errors.length)throw new Error(errors.join(';'));
    return {passed:true,before,after,reports,errors};
  } finally {
    await page.unroute('**/api/sim/sync');
    await page.unroute('**/api/plans/visual-step');
  }
}

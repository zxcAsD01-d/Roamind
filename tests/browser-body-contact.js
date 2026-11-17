async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(()=>document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#power-toggle').check();
  await page.locator('#touch-partner').click();
  await page.locator('#ego-fov').fill('105'); await page.locator('#ego-fov').blur();
  await page.locator('#ego-pitch').fill('15');
  const clientId=await page.locator('#viewport').getAttribute('data-client-id');
  const command=await page.evaluate(async clientId=>{
    const r=await fetch('http://127.0.0.1:8100/api/sim/move',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clientId,destination:[-5,.92,3.48]})});return r.json();
  },clientId);
  for(let attempt=0;attempt<200;attempt++) {
    const status=await page.evaluate(async id=>(await fetch('http://127.0.0.1:8100/api/sim/commands/'+id)).json(),command.id);
    if(status.status==='completed')break;
    if(['failed','interrupted','superseded'].includes(status.status))throw new Error(status.detail);
    if(attempt===199)throw new Error('摆位超时');
    await page.waitForTimeout(100);
  }
  // Keep high-rate vision until the contact starts, independent of UI pointer timing.
  await page.locator('#power-toggle').uncheck();
  await page.locator('#ego-toggle').check();
  await page.waitForFunction(()=>[...document.querySelector('#touch-person').options].some(o=>o.value),null,{timeout:20000});
  await page.locator('#ego-preview').screenshot({path:'output/playwright/touch-ocr/body-view.png'});
  return {clientId};
}

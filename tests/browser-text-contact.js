async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#power-toggle').check();
  await page.locator('#ego-fov').fill('95'); await page.locator('#ego-fov').blur();
  await page.locator('#ego-pitch').fill('30');
  await page.locator('#ocr-board').click();
  await page.locator('#ego-toggle').check();
  await page.waitForFunction(()=>!!document.querySelector('#depth-status').dataset.result);
  await page.locator('#ocr-scan').click();
  await page.waitForFunction(()=>!!document.querySelector('#ocr-status').dataset.result,null,{timeout:15000});
  const scan=await page.locator('#ocr-status').evaluate(e=>JSON.parse(e.dataset.result));
  await page.locator('#ego-preview').screenshot({path:'output/playwright/touch-ocr/text-view.png'});
  if (!scan.selected) throw new Error(`OCR 文字未唯一匹配 ${JSON.stringify(scan)}`);
  const clientId=await page.locator('#viewport').getAttribute('data-client-id');
  const command=await page.evaluate(async clientId=>{
    const response=await fetch('http://127.0.0.1:8100/api/sim/text-contact',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({clientId,text:'确认',direction:'anywhere'})});
    const result=await response.json();if(!response.ok)throw new Error(JSON.stringify(result));return result;
  },clientId);
  const samples=[];
  for(let index=0;index<450;index++) {
    const sample=await page.evaluate(async id=>{
      const response=await fetch('http://127.0.0.1:8100/api/sim/commands/'+id);return {
        command:await response.json(), contact:JSON.parse(document.querySelector('#contact-live').dataset.tracking||'null'),
        ocr:document.querySelector('#ocr-status').textContent, touch:document.querySelector('#touch-status').textContent};
    },command.id);
    if(sample.contact && (!samples.length || sample.contact.frame!==samples.at(-1).frame))samples.push(sample.contact);
    if(['completed','failed','interrupted','superseded'].includes(sample.command.status)) {
      const result={scan,clientId,command:sample.command,samples,status:sample.ocr,touch:sample.touch};
      if(sample.command.status!=='completed' || !samples.some(item=>item.contactSamples>0 && item.reachedSamples>=2))
        throw new Error(`文字接触验收失败 ${JSON.stringify(result)}`);
      return result;
    }
    await page.waitForTimeout(100);
  }
  throw new Error('文字触碰45秒未结束');
}

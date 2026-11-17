async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(()=>document.querySelector('#viewport')?.dataset.clientId);
  await page.locator('#ambient-idle').uncheck();
  await page.locator('.left-panel > .research-controls > summary').click();
  await page.locator('#ego-fov').fill('95'); await page.locator('#ego-fov').blur();
  await page.locator('#ego-pitch').fill('30');
  await page.locator('.runtime-interactions > summary').click();
  await page.locator('#ocr-board').click();
  await page.locator('#ocr-text').fill('确认');
  await page.locator('#ocr-touch').click();
  await page.waitForFunction(()=>JSON.parse(document.querySelector('#touch-status').dataset.target||'null')?.targetId.startsWith('text-'));
  await page.locator('#goal-kind').selectOption('contact');
  await page.locator('#goal-start').click();
  const samples=[];
  for(let n=0;n<500;n++) {
    const result=await page.locator('#goal-status').evaluate(e=>JSON.parse(e.dataset.goal||'null'));
    const contact=await page.locator('#contact-live').evaluate(e=>JSON.parse(e.dataset.tracking||'null'));
    if(contact&&(!samples.length||contact.frame!==samples.at(-1).frame))samples.push(contact);
    if(result&&['completed','failed','cancelled'].includes(result.status)) {
      if(result.status!=='completed'||!samples.some(s=>s.reachedSamples>=2))throw new Error('任务文字接触失败 '+JSON.stringify({result,samples}));
      await page.screenshot({path:'output/playwright/character/text-goal.png'});
      return {goal:result,samples,status:await page.locator('#ocr-status').textContent()};
    }
    await page.waitForTimeout(100);
  }
  throw new Error('文字接触任务超时');
}

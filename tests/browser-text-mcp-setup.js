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
  const tracking=[];
  page.on('response', async response=>{
    if(response.url().endsWith('/api/touch/track'))tracking.push({at:Date.now()/1000,data:await response.json()});
  });
  for(let i=0;i<450;i++) {
    await page.evaluate(()=>document.querySelector('#contact-live').dataset.tracking);
    await page.waitForTimeout(100);
  }
  return {clientId,trackingResponses:tracking,power:await page.locator("#power-toggle").isChecked(),tracking:await page.locator("#contact-live").getAttribute("data-tracking")};
}

async (page) => {
  const results=[];
  for(let i=0;i<5;i++) {
    await page.goto('http://127.0.0.1:5173/');
    await page.waitForFunction(()=>document.querySelector('#load-status')?.textContent.includes('步幅比例'));
    await page.locator('#power-toggle').check();
    await page.locator('#ego-toggle').check();
    await page.waitForFunction(()=>!!document.querySelector('#depth-status').dataset.result);
    const state=await page.evaluate(async()=>{
      const response=await fetch('http://127.0.0.1:8200/health');return response.json();
    });
    results.push(state.stereoSessions);
    if(state.stereoSessions.length>2)throw new Error(`旧页面双目会话未释放 ${JSON.stringify(results)}`);
  }
  return {reloads:5,sessions:results,power:await page.locator('#power-toggle').isChecked()};
}

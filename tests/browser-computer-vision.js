async page => {
 await page.route('**/api/world',async route=>{const response=await route.fetch(),world=await response.json();world.spawn=[8,.92,-4.45];await route.fulfill({response,json:world});});
 await page.goto('http://127.0.0.1:5173/');
 await page.waitForFunction(()=>document.querySelector('#load-status')?.textContent.includes('步幅比例'),null,{timeout:60000});
 await page.locator('#ambient-idle').uncheck();
 await page.evaluate(()=>{const el=document.querySelector('#ego-toggle');if(el.checked)el.click();});
 const request=async(path,body={})=>page.evaluate(async({path,body})=>{const clientId=document.querySelector('#viewport').dataset.clientId;
 const response=await fetch('http://127.0.0.1:8100'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clientId,...body})});
 const data=await response.json();if(!response.ok)throw new Error(JSON.stringify(data));return data;},{path,body});
 await page.waitForTimeout(1200);
 const command=await request('/api/sim/face',{heading:180,duration:2});
 for(let i=0;i<50;i++){const c=await page.evaluate(async id=>await(await fetch('http://127.0.0.1:8100/api/sim/commands/'+id)).json(),command.id);if(c.status==='completed')break;await page.waitForTimeout(200);}
 await page.evaluate(()=>{document.querySelector('#vision-classes').value='chair,keyboard,computer monitor,table,wall,floor';const pitch=document.querySelector('#ego-pitch');pitch.value='20';pitch.dispatchEvent(new Event('change',{bubbles:true}));});
 await page.evaluate(()=>{const el=document.querySelector('#ego-toggle');if(!el.checked)el.click();});
 const samples=[];let chosen=null;
 for(let i=0;i<24;i++){await page.waitForTimeout(1000);const ctx=await request('/api/agent/context');
 samples.push({objects:ctx.interactions.objects,visualStatus:await page.locator('#vision-status').textContent(),depth:await page.locator('#depth-status').textContent()});
 chosen=ctx.interactions.objects.find(row=>row.options.some(option=>option.operation==='play_computer'&&option.enabled));if(chosen)break;}
 await page.screenshot({path:'output/playwright/computer-room/vision-test.png'});
 await page.evaluate(()=>{const el=document.querySelector('#ego-toggle');if(el.checked)el.click();});await page.unroute('**/api/world');
 return {fixture:'Explicit test spawn only; observations from rendered stereo/YOLO',visualNodesReady:!!chosen,chosen,samples};
}

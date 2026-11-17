async page=>{
 await page.goto('http://127.0.0.1:5173/');
 await page.waitForFunction(()=>document.querySelector('#load-status')?.textContent.includes('步幅比例'),null,{timeout:60000});
 await page.waitForFunction(()=>Number(document.querySelector('#need-energy').value)>0);
 await page.locator('#body-stand').click();
 await page.waitForFunction(()=>{const raw=document.querySelector('#goal-status').dataset.goal;return raw&&JSON.parse(raw)?.status==='completed';},null,{timeout:15000});
 const result=await page.evaluate(async()=>{
 const clientId=document.querySelector('#viewport').dataset.clientId;
 const response=await fetch('http://127.0.0.1:8100/api/agent/context',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clientId})});
 const ctx=await response.json();
 if(ctx.goal.status!=='completed'||ctx.execution.interaction.posture!=='stand')throw new Error('body step completion missing');
 if(!ctx.kernel.needs||!ctx.capabilities.nodeOperations.pickup)throw new Error('needs/node API unavailable');
 const input=document.querySelector('#ego-toggle');if(input.checked)input.click();
 return {goal:ctx.goal.status,needs:ctx.kernel.needs,body:ctx.execution.interaction,capabilities:ctx.capabilities.version};
 });
 await page.screenshot({path:'output/needs-interaction/frontend-final.png'});return result;
}

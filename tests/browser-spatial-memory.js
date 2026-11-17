async page => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const fixture = await (await page.request.get('http://127.0.0.1:5173/output/spatial-memory/fixture.json')).json();
  const navigation = await (await page.request.get('http://127.0.0.1:5173/output/spatial-memory/navigation-fixture.json')).json();
  const backend = await page.evaluate(async navigation => {
    const clientId = 'spatial-memory-browser-regression';
    const post = async (path, body) => {
      const response = await fetch('http://127.0.0.1:8100'+path, {method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({clientId,...body})});
      const result = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(result));
      return result;
    };
    await post('/api/sim/autonomy/reset', {});
    let state;
    for (let index=0;index<4;index++) {
      const stamp=Date.now()/1000;
      const pose=[1,0,0,1.75,0,1,0,0,0,0,1,1.75,0,0,0,1];
      state=await post('/api/sim/sync', {state:{},vision:{observedAt:stamp,semanticReady:false,
        signature:Array(64).fill(.2),stereo:{observedAt:stamp,mapping:true,pose:{cameraToWorld:pose},
          sectors:[],navigationGrid:navigation}}});
      if (index<3) await new Promise(resolve=>setTimeout(resolve,400));
    }
    const memory=state.behavior.spatialMemory;
    if (memory.topology.regions.length!==2 || memory.topology.exits.length!==1
        || memory.topology.components!==1) throw new Error(JSON.stringify(memory.topology));
    return {cells:memory.knownCells.length,regions:memory.topology.regions.length,
      exits:memory.topology.exits,components:memory.topology.components};
  },navigation);
  const result = await page.evaluate(async memory => {
    const {MemoryView}=await import('/src/memory_view.ts');
    const originalText=document.querySelector('#character-memory');
    const originalCanvas=document.querySelector('#memory-grid');
    originalText.id='memory-saved-text'; originalCanvas.id='memory-saved-canvas';
    const panel=document.createElement('section');
    panel.style.cssText='position:fixed;right:20px;top:20px;z-index:999999;background:#111a1b;padding:20px;color:white;width:560px;font:14px/1.6 sans-serif';
    panel.innerHTML='<p id="character-memory" style="white-space:pre-line"></p><canvas id="memory-grid" width="560" height="340" style="width:560px;height:340px"></canvas><p>底色：区域 · 实线：实测边界 · 灰虚线：未知 · 绿点：可通行出口</p>';
    document.body.append(panel);
    const view=new MemoryView();
    view.update(memory);
    const canvas=panel.querySelector('canvas');
    const first=canvas.toDataURL();
    const description=panel.querySelector('p').textContent;
    if (!description.includes('与其他区域联通') || !description.includes('联通 2 区')) throw new Error(description);
    const points=canvas.getContext('2d').getImageData(0,0,560,340).data;
    const colored=new Set();
    for (let i=0;i<points.length;i+=4) colored.add(`${points[i]},${points[i+1]},${points[i+2]}`);
    if (colored.size<8) throw new Error('区域/边界没有渲染');
    const changed=structuredClone(memory);
    changed.topology.revision++;
    changed.topology.exits[0].state='blocked';
    changed.topology.exits[0].traversable=false;
    changed.topology.exits[0].connectsRegions=false;
    changed.topology.regions.forEach(region=>{region.enclosure.status='observed-enclosed';region.enclosure.closedBoundary=true;region.reachableRegionCount=1;});
    view.update(changed);
    const redrawn=first!==canvas.toDataURL();
    const closed=panel.querySelector('p').textContent;
    if (!redrawn || !closed.includes('已观测地板边界闭合') || !closed.includes('1 受阻')) throw new Error(closed);
    const rect=canvas.getBoundingClientRect();
    const projected=view.positionAt({currentTarget:canvas,clientX:rect.left+rect.width/2,clientY:rect.top+rect.height/2});
    if (!projected.every(Number.isFinite)) throw new Error('点击投影无效');
    memory.topology.revision+=2;view.update(memory);
    window.spatialMemoryCleanup=()=>{panel.remove();originalText.id='character-memory';originalCanvas.id='memory-grid';};
    return {description,closed,redrawn,colors:colored.size,projected};
  },fixture);
  await page.locator('section').filter({has:page.locator('#character-memory')}).last().screenshot({path:'output/playwright/spatial-memory.png'});
  await page.evaluate(()=>window.spatialMemoryCleanup());
  if (errors.length) throw new Error(JSON.stringify(errors));
  return {backend,view:result,errors};
}

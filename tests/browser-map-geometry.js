async page => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.unroute('http://127.0.0.1:8200/**');
  await page.reload();
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'), null, {timeout:60000});
  await page.locator('details.research-controls').evaluateAll(items => items.forEach(item => { item.open = true; }));
  await page.locator('#ambient-idle').uncheck();
  await page.locator('#ego-toggle').check();
  await page.waitForFunction(() => {
    const state = JSON.parse(document.querySelector('#world-status')?.dataset.state || 'null');
    return state?.surfaces > 0 && state.volumes > 0;
  }, null, {timeout:45000});
  await page.waitForFunction(() => document.querySelector('#ego-preview')?.dataset.semanticFrameId, null, {timeout:30000});
  const frame = await page.locator('#ego-preview').evaluate(canvas => ({
    displayed:canvas.dataset.frameId, semantic:canvas.dataset.semanticFrameId,
  }));
  if (frame.displayed !== frame.semantic) throw new Error('语义图像帧不一致');
  const buffers = await page.evaluate(async () => {
    const {VisualWorldMap} = await import('/src/worldmap.ts');
    const canvas = document.createElement('canvas');
    canvas.style.cssText='width:640px;height:460px;position:fixed;left:0;top:0;z-index:99999';
    document.body.append(canvas);
    const view = new VisualWorldMap(canvas);
    const surface = {id:'surface-a',volumeId:'volume-a',normal:[0,0,1],centroid:[0,1,2],
      bounds:[[-1,0,2],[1,2,2]],vertices:[-1,0,2,1,0,2,1,2,2,-1,2,2],indices:[0,1,2,0,2,3],
      color:0x7896b4,meanColor:[120,150,180],semantic:1,label:'wall',confidence:.9,
      measuredRatio:1,voxelCount:100,continuous:true};
    const volume = {id:'volume-a',label:'wall',semantic:1,semanticConfidence:.9,instanceId:'object-a',
      centroid:[0,1,2],bounds:[[-1,0,1.96],[1,2,2.04]],voxelCount:100,measuredRatio:1,
      surfaceIds:['surface-a'],lastSeen:100,geometry:'observed-surface-volume'};
    const point = {mapId:'a',positions:[0,1,2],colors:[0x7896b4],semantics:[1],labels:['unlabeled','wall'],
      objects:[],totalPoints:1,inferred:[false],surfaces:[surface],volumes:[volume]};
    view.updateMap({...point,revision:1,geometryRevision:1});
    const cloud = view.geometry.getAttribute('position');
    const surfaceBuffer = view.surfaceGeometry.getAttribute('position');
    const volumeBuffer = view.volumeGeometry.getAttribute('position');
    const firstVersion = surfaceBuffer.version;
    for (let index=2;index<=100;index++) view.updateMap({...point,revision:index,geometryRevision:1});
    const cachedGeometry = surfaceBuffer.version === firstVersion;
    view.updateMap({...point,revision:101,geometryRevision:2});
    const reused = cloud === view.geometry.getAttribute('position')
      && surfaceBuffer === view.surfaceGeometry.getAttribute('position')
      && volumeBuffer === view.volumeGeometry.getAttribute('position');
    const counts = {surface:view.surfaceGeometry.drawRange.count,volume:view.volumeGeometry.drawRange.count};
    view.updateMap({...point,revision:99,geometryRevision:3,positions:[5,1,2]});
    const stayedLatest = cloud.getX(0) === 0;
    view.updateMap({...point,mapId:'b',revision:1,geometryRevision:1,positions:[2,1,2]});
    view.updateMap({...point,mapId:'a',revision:200,geometryRevision:200});
    const retiredRejected = cloud.getX(0) === 2;
    const stableGeometries = view.renderer.info.memory.geometries;
    view.updateMap({...point,mapId:'b',revision:2,geometryRevision:2,surfaces:[],volumes:[],positions:[],colors:[]});
    const cleared = view.surfaceGeometry.drawRange.count === 0 && view.volumeGeometry.drawRange.count === 0;
    view.resetView();
    const resetAccepted = view.geometry.drawRange.count === 0;
    view.updateMap({...point,mapId:'a',revision:1,geometryRevision:1});
    const restarted = view.geometry.drawRange.count === 1;
    view.controls.dispose(); view.renderer.dispose(); canvas.remove();
    return {cachedGeometry,reused,counts,stayedLatest,retiredRejected,stableGeometries,cleared,resetAccepted,restarted};
  });
  for (const key of ['cachedGeometry','reused','stayedLatest','retiredRejected','cleared','resetAccepted','restarted']) {
    if (!buffers[key]) throw new Error(JSON.stringify(buffers));
  }
  if (buffers.counts.surface !== 6 || buffers.counts.volume !== 24) throw new Error(JSON.stringify(buffers));
  const state = await page.locator('#world-status').evaluate(node => JSON.parse(node.dataset.state));
  const snapshot = await page.evaluate(async client => {
    const response = await fetch(`http://127.0.0.1:8200/api/stereo/state?clientId=${encodeURIComponent(client)}`);
    return response.json();
  }, state.clientId);
  const map = snapshot.map;
  if (!map?.surfaces?.length || !map?.volumes?.length) throw new Error('真实双目未生成几何');
  const geometry = {points:map.totalPoints,surfaces:map.surfaces.length,volumes:map.volumes.length,
    semanticVolumes:map.volumes.filter(v=>v.semantic).length,quads:map.surfaces.reduce((n,s)=>n+s.indices.length/6,0),
    metrics:map.geometry,geometryRevision:map.geometryRevision};
  for (const surface of map.surfaces) {
    if (surface.vertices.some(v=>!Number.isFinite(v)) || surface.indices.some(i=>i*3>=surface.vertices.length)) {
      throw new Error('无效表面顶点或索引');
    }
  }
  if (errors.length) throw new Error(JSON.stringify(errors));
  await page.locator('#world-cloud').evaluate(canvas => {
    for (let parent=canvas.parentElement;parent;parent=parent.parentElement) {
      if (parent instanceof HTMLDetailsElement) parent.open=true;
    }
  });
  await page.locator('#world-cloud').screenshot({path:'output/playwright/map-reconstruction.png'});
  await page.locator('#ego-toggle').uncheck();
  return {frame,buffers,geometry,errors};
}

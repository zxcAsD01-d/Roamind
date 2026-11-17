async page => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#ambient-idle').uncheck();
  await page.locator('#ego-toggle').evaluate(node => { node.checked = false; node.dispatchEvent(new Event('change', {bubbles:true})); });
  const result = await page.evaluate(async () => {
    const THREE = await import('/node_modules/.vite/deps/three.js');
    const { GLTFLoader } = await import('/node_modules/.vite/deps/three_addons_loaders_GLTFLoader__js.js');
    const { VRMLoaderPlugin, VRMUtils } = await import('/node_modules/.vite/deps/@pixiv_three-vrm.js');
    const { SmplClip, VrmRetargeter, measureVrmRest } = await import('/src/retarget.ts');
    const { LocomotionLibrary } = await import('/src/locomotion.ts');
    const { LumenMotionSystem } = await import('/src/lumen/runtime.ts');
    const loader = new GLTFLoader(); loader.register(parser => new VRMLoaderPlugin(parser));
    const gltf = await loader.loadAsync('/models/Wolferia.vrm');
    const vrm = gltf.userData.vrm, group = new THREE.Group(); group.add(vrm.scene); vrm.scene.rotation.y = Math.PI;
    const bounds = new THREE.Box3().setFromObject(vrm.scene), center = bounds.getCenter(new THREE.Vector3());
    vrm.scene.position.set(-center.x, -bounds.min.y, -center.z); measureVrmRest(vrm);
    const library = await LocomotionLibrary.load(vrm), lumen = await LumenMotionSystem.load(vrm);
    const seed = library.idle(new THREE.Vector3(), 0);
    const route = library.compile({id:'lumen-walk',duration:15,timing:'flexible',path:[[0,0,0],[0,0,7]],
      timeline:[{action:'forward',start:0,end:15,distance:7,continuous:true}]},seed,()=>null);
    const clip = SmplClip.from(route.motion), driver = new VrmRetargeter(vrm,clip);
    const [wave, raise, point, left] = await Promise.all(['wave','raise','point','lumen:left-向上抬高'].map(action => lumen.prepare(action,3)));
    let maximumArmStep = 0, maximumLegDeviation = 0, maximumRootDeviation = 0, peakArmDifference = 0, previous = null;
    let completed = false; const reports = [], timings = [];
    const names = ['hips','spine','chest','leftUpperLeg','leftLowerLeg','leftFoot','leftToes','rightUpperLeg','rightLowerLeg','rightFoot','rightToes'];
    for (let i=0;i<1000;i++) {
      const dt = 1/60, time = i*dt;
      lumen.restore(); group.position.copy(driver.rootAt(Math.min(clip.frames-1,time*20)));
      driver.apply(Math.min(clip.frames-1,time*20));
      const base = names.map(name=>vrm.humanoid.getNormalizedBoneNode(name).quaternion.clone());
      const armBase = vrm.humanoid.getNormalizedBoneNode('rightUpperArm').quaternion.clone();
      const root = group.position.clone();
      if(i===0)lumen.start('wave',wave);
      if(i===60)lumen.start('raise',raise);
      if(i===130)lumen.start('point',point);
      if(i===150)lumen.start('left-independent',left);
      const begin=performance.now();lumen.update(dt);timings.push(performance.now()-begin);
      names.forEach((name,j)=>{maximumLegDeviation=Math.max(maximumLegDeviation,...base[j].toArray().map((v,k)=>Math.abs(v-vrm.humanoid.getNormalizedBoneNode(name).quaternion.toArray()[k])));});
      maximumRootDeviation=Math.max(maximumRootDeviation,root.distanceTo(group.position));
      const arm=vrm.humanoid.getNormalizedBoneNode('rightUpperArm').quaternion.clone();
      peakArmDifference=Math.max(peakArmDifference,arm.clone().normalize().angleTo(armBase.normalize()));
      if(previous)maximumArmStep=Math.max(maximumArmStep,arm.clone().normalize().angleTo(previous.normalize()));previous=arm;
      reports.push(...lumen.drainReports());
      if(i>130&&!lumen.active){completed=true;break;}
    }
    if(!completed)throw new Error('流明动作未结束 '+JSON.stringify(lumen.state()));
    if(reports.some(row=>row.status==='failed'))throw new Error(JSON.stringify(reports));
    if(!reports.some(row=>row.id==='point'&&row.status==='completed'))throw new Error('末动作未实际完成');
    if(!reports.some(row=>row.id==='left-independent'&&row.status==='completed'))throw new Error('独立左臂轨道未完成');
    if(maximumLegDeviation>1e-6||maximumRootDeviation>1e-8)throw new Error('上身叠加改变了行走与根运动 '+JSON.stringify({maximumLegDeviation,maximumRootDeviation}));
    if(peakArmDifference<.05)throw new Error('流明动作没有实际改变手臂');
    if(maximumArmStep>.16)throw new Error('手臂每帧跳变过大 '+maximumArmStep);
    if(reports.filter(row=>row.status==='interrupted').length!==2)throw new Error('重叠切换没有报告旧动作');
    timings.sort((a,b)=>a-b);VRMUtils.deepDispose(vrm.scene);
    return {reports,maximumArmStep,maximumLegDeviation,maximumRootDeviation,peakArmDifference,
      p95Ms:timings[Math.floor(timings.length*.95)],completed,engine:lumen.state().engine};
  });
  const clientId=await page.locator('#viewport').getAttribute('data-client-id');
  const command=await page.evaluate(async clientId=>{
    const response=await fetch('http://127.0.0.1:8100/api/sim/action',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({clientId,action:'wave',duration:3})});
    const body=await response.json();if(!response.ok)throw new Error(JSON.stringify(body));return body;
  },clientId);
  let final;
  for(let i=0;i<150;i++){
    final=await page.evaluate(async id=>(await fetch('http://127.0.0.1:8100/api/sim/commands/'+id)).json(),command.id);
    if(['completed','failed','interrupted','superseded'].includes(final.status))break;
    await page.waitForTimeout(100);
  }
  if(final.status!=='completed'||final.kind!=='gesture')throw new Error('前端流明命令失败 '+JSON.stringify(final));
  const parallel=await page.evaluate(async clientId=>Promise.all(['wave','lumen:left-向上抬高'].map(async action=>{
    const response=await fetch('http://127.0.0.1:8100/api/sim/action',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({clientId,action,duration:3})});
    const value=await response.json();if(!response.ok)throw new Error(JSON.stringify(value));return value;
  })),clientId);
  let parallelFinal=[];
  for(let i=0;i<150;i++){
    parallelFinal=await page.evaluate(async ids=>Promise.all(ids.map(async id=>(await fetch('http://127.0.0.1:8100/api/sim/commands/'+id)).json())),parallel.map(c=>c.id));
    if(parallelFinal.every(c=>c.status==='completed'))break;
    if(parallelFinal.some(c=>['failed','interrupted','superseded'].includes(c.status)))throw new Error('双臂准备或播放互相取消 '+JSON.stringify(parallelFinal));
    await page.waitForTimeout(100);
  }
  if(!parallelFinal.every(c=>c.status==='completed'))throw new Error('双臂并行命令未完成');
  if(errors.length)throw new Error(errors.join(';'));
  await page.screenshot({path:'output/lumen-rebuild/runtime.png'});
  return {passed:true,...result,command:final,parallel:parallelFinal,microphoneClosed:await page.locator('#mic-stop').isDisabled(),errors};
}

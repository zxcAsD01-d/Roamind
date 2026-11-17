async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'), { timeout: 60000 });
  return page.evaluate(async () => {
    const THREE = await import('/node_modules/.vite/deps/three.js');
    const { GLTFLoader } = await import('/node_modules/.vite/deps/three_addons_loaders_GLTFLoader__js.js');
    const { VRMLoaderPlugin, VRMUtils } = await import('/node_modules/.vite/deps/@pixiv_three-vrm.js');
    const { SmplClip, measureVrmRest } = await import('/src/retarget.ts');
    const { LocomotionLibrary, presetSeed } = await import('/src/locomotion.ts');
    const { tableBoundary, scaledBoundary } = await import('/src/motion_table_features.ts');
    const { searchMotionTable } = await import('/src/motion_table_search.ts');
    const loader = new GLTFLoader(); loader.register(parser => new VRMLoaderPlugin(parser));
    const gltf = await loader.loadAsync('/models/Wolferia.vrm'), vrm = gltf.userData.vrm;
    const group = new THREE.Group(); group.add(vrm.scene); vrm.scene.rotation.y = Math.PI;
    const bounds = new THREE.Box3().setFromObject(vrm.scene), center = bounds.getCenter(new THREE.Vector3());
    vrm.scene.position.set(-center.x, -bounds.min.y, -center.z); measureVrmRest(vrm);
    const started = performance.now(), library = await LocomotionLibrary.load(vrm);
    const seed = library.idle(new THREE.Vector3(), 0);
    const curve = (radius, angle, sign = 1) => Array.from({ length: 41 }, (_, i) => {
      const t = i/40*angle; return [sign*radius*(1-Math.cos(t)), 0, radius*Math.sin(t)];
    });
    const cases = { straight: [[0,0,0],[0,0,3]], rightArc: curve(3,1.1), leftArc: curve(3,1.1,-1),
      gentle: curve(5,.8), sCurve: Array.from({length:41},(_,i)=>{const t=i/40;return [.4*(1-Math.cos(2*Math.PI*t)),0,4*t];}) };
    const results = { entries: library.table.document.entries.length, loadMs: performance.now()-started, cases: {} };
    let featureError = 0;
    for (const asset of library.table.assets.slice(0, 80)) {
      const actual = scaledBoundary(tableBoundary(asset.clip, asset.end),asset.scale,asset.rate);
      featureError = Math.max(featureError,...actual.map((v,i)=>Math.abs(v-asset.last[i])));
    }
    results.featureError = featureError;
    for (const [name,path] of Object.entries(cases)) {
      const begin = performance.now();
      try {
        const result = library.compileTable(name,path,seed,()=>null), clip = SmplClip.from(result.motion);
        let jointStep = 0, seamSpeedMin = Infinity, seamSpeedMax = 0;
        for (let f=seed.motion.frames;f<clip.frames;f++) for(let j=0;j<27;j++) {
          const a = new THREE.Quaternion().fromArray(clip.local,((f-1)*27+j)*4);
          jointStep = Math.max(jointStep,a.angleTo(new THREE.Quaternion().fromArray(clip.local,(f*27+j)*4)));
        }
        for(const phase of result.phases.filter(p=>p.kind==='walk')) {
          const f=Math.round(phase.start*20),a=clip.rootAt(f-1,new THREE.Vector3()),b=clip.rootAt(f,new THREE.Vector3());
          const speed=Math.hypot(a.x-b.x,a.z-b.z)*20;seamSpeedMin=Math.min(seamSpeedMin,speed);seamSpeedMax=Math.max(seamSpeedMax,speed);
        }
        const prefixUnchanged = seed.motion.localRot.every((v,i)=>v===result.motion.localRot[i])&&seed.motion.rootDelta.every((v,i)=>v===result.motion.rootDelta[i]);
        results.cases[name] = { ok:true,...result.table, compileMs:performance.now()-begin,quality:library.lastQuality,
          seconds:result.seconds,jointStep,seamSpeedMin,seamSpeedMax,prefixUnchanged,phases:result.phases.map(p=>p.kind) };
        if(name==='rightArc') {
          const at=Math.min(3,result.seconds/2),nextSeed=presetSeed(clip,at,true,new THREE.Vector3());nextSeed.phase='walk';
          const end=nextSeed.motion.rootDelta.slice(-3),heading=new THREE.Vector3(0,0,1).applyQuaternion(new THREE.Quaternion().fromArray(SmplClip.from(nextSeed.motion).local,(nextSeed.motion.frames-1)*27*4));
          const yaw=Math.atan2(heading.x,heading.z),newPath=curve(4,.7).map(row=>new THREE.Vector3(...row).applyAxisAngle(new THREE.Vector3(0,1,0),yaw).add(new THREE.Vector3(...end)).toArray());
          try {
            const next=library.compileTable('continuation',newPath,nextSeed,()=>null);
            results.continuation={ok:true,...next.table,prefixUnchanged:nextSeed.motion.localRot.every((v,i)=>next.motion.localRot[i]===v)&&nextSeed.motion.rootDelta.every((v,i)=>next.motion.rootDelta[i]===v),phases:next.phases.map(p=>p.kind)};
          } catch(error) {results.continuation={ok:false,error:error.message};}
        }
      } catch(error) {
        results.cases[name]={ok:false,error:error.message,compileMs:performance.now()-begin};
        try {const route=searchMotionTable(library.table.assets,path,seed);results.cases[name].search=route.diagnostics;}catch(searchError){results.cases[name].searchError=searchError.message;}
      }
    }
    let collisionRejected=false;
    try {library.compileTable('blocked',cases.rightArc,seed,()=> 'test wall');}catch(error){collisionRejected=error.constructor.name==='PresetCollisionError';}
    results.collisionRejected=collisionRejected;
    if (featureError > 1e-4 || !collisionRejected) throw new Error('特征一致性或碰撞门限回归');
    for (const [name,value] of Object.entries(results.cases)) {
      if (!value.ok) throw new Error(`${name} 未通过动作表：${value.error}`);
      if (!value.prefixUnchanged || value.goalError > .12 || value.pathMaxError > .3
        || value.quality.mean > .12 || value.quality.p95 > .32 || value.jointStep > .4
        || value.seamVelocityMax > .181 || value.phases.filter(p=>p==='stop').length!==1
        || value.phases.filter(p=>p==='start').length>1 || value.phases.includes('idle'))
        throw new Error(`${name} 连贯性指标不合格：${JSON.stringify(value)}`);
    }
    const integrated=library.compile({id:'integrated-curve',duration:10,timing:'flexible',path:cases.rightArc,
      timeline:[{action:'forward',start:0,end:10,distance:3.3,continuous:true}]},seed,()=>null);
    if(!integrated.table) throw new Error('前端规划没有接入动作表');
    results.integrated=true;
    results.gestures=[];
    const prompts=[...new Set(library.table.document.entries.filter(e=>e.role==='action').map(e=>e.prompt))];
    for(const prompt of prompts){
      const motion=library.tableAction(prompt,seed,()=>null);
      if(motion.seconds<2||!seed.motion.localRot.every((v,i)=>motion.motion.localRot[i]===v)) throw new Error('完整手势或前缀回归');
      results.gestures.push({prompt,seconds:motion.seconds,phases:motion.phases.map(p=>p.kind)});
    }
    window.motionTableTest={library,vrm,THREE,seed,cases};
    return results;
  });
}

async page => page.evaluate(async () => {
  const THREE = await import('/node_modules/.vite/deps/three.js');
  const { GLTFLoader } = await import('/node_modules/.vite/deps/three_addons_loaders_GLTFLoader__js.js');
  const { VRMLoaderPlugin, VRMUtils } = await import('/node_modules/.vite/deps/@pixiv_three-vrm.js');
  const { LumenMotionSystem } = await import('/src/lumen/runtime.ts');
  const { LocomotionLibrary } = await import('/src/locomotion.ts');
  const { SmplClip, VrmRetargeter, measureVrmRest } = await import('/src/retarget.ts');
  const loader=new GLTFLoader();loader.register(parser=>new VRMLoaderPlugin(parser));
  const gltf=await loader.loadAsync('/models/Wolferia.vrm'),vrm=gltf.userData.vrm;
  vrm.scene.rotation.y=Math.PI;measureVrmRest(vrm);
  const library=await LocomotionLibrary.load(vrm),system=await LumenMotionSystem.load(vrm);
  const driver=new VrmRetargeter(vrm,SmplClip.from(library.idle(new THREE.Vector3(),0).motion));
  const response=await fetch('http://127.0.0.1:8100/api/actions/capabilities');
  if(!response.ok)throw new Error('动作目录不可读');
  const catalog=await response.json(),results=[];
  const protectedNames=['hips','spine','chest','leftUpperLeg','leftLowerLeg','leftFoot','rightUpperLeg','rightLowerLeg','rightFoot'];
  try {
    for(const row of catalog.actions){
      const prepared=await system.prepare(row.action,3);
      system.restore();driver.apply(0);system.start(row.action,prepared);
      let maximumStep=0,previous=null,frames=0;const reports=[];
      const armNames=['leftUpperArm','leftLowerArm','leftHand','rightUpperArm','rightLowerArm','rightHand'];
      for(;frames<900&&system.active;frames++){
        system.restore();driver.apply(0);
        const before=protectedNames.map(name=>vrm.humanoid.getNormalizedBoneNode(name).quaternion.toArray());
        system.update(1/60);
        protectedNames.forEach((name,i)=>{if(before[i].some((v,j)=>v!==vrm.humanoid.getNormalizedBoneNode(name).quaternion.toArray()[j]))
          throw new Error('动作修改了根或腿 '+row.action);});
        const arm=armNames.map(name=>vrm.humanoid.getNormalizedBoneNode(name).quaternion.clone().normalize());
        if(previous)maximumStep=Math.max(maximumStep,...arm.map((q,i)=>q.angleTo(previous[i])));previous=arm;
        reports.push(...system.drainReports());
      }
      if(system.active||reports.some(r=>r.status==='failed')||!reports.some(r=>r.status==='completed'))
        throw new Error('原生目录动作未完成 '+JSON.stringify({action:row.action,reports}));
      if(maximumStep>.16)throw new Error('原生目录动作跳变 '+row.action+': '+maximumStep);
      results.push({action:row.action,frames,maximumStep,reports});
    }
  } finally {system.cancel();VRMUtils.deepDispose(vrm.scene);}
  return {passed:true,actions:results.length,results};
})

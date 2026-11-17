async page => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'), null, {timeout:60000});
  await page.evaluate(()=>{ const input=document.querySelector('#ego-toggle');if(input.checked)input.click(); });
  return await page.evaluate(async () => {
    const THREE = await import('/node_modules/.vite/deps/three.js');
    const { GLTFLoader } = await import('/node_modules/.vite/deps/three_addons_loaders_GLTFLoader__js.js');
    const { VRMLoaderPlugin } = await import('/node_modules/.vite/deps/@pixiv_three-vrm.js');
    const { SmplClip, VrmRetargeter, measureVrmRest } = await import('/src/retarget.ts');
    const { LocomotionLibrary } = await import('/src/locomotion.ts');
    const { InteractionBody } = await import('/src/interaction_body.ts');
    const { InteractionProps } = await import('/src/interaction_props.ts');
    const loader = new GLTFLoader(); loader.register(parser => new VRMLoaderPlugin(parser));
    const vrm = (await loader.loadAsync('/models/Wolferia.vrm')).userData.vrm;
    const scene = new THREE.Scene(), root = new THREE.Group(); scene.add(root); root.add(vrm.scene); vrm.scene.rotation.y=Math.PI;
    const bounds = new THREE.Box3().setFromObject(vrm.scene), center=bounds.getCenter(new THREE.Vector3());
    vrm.scene.position.set(-center.x,-bounds.min.y,-center.z); measureVrmRest(vrm);
    const library=await LocomotionLibrary.load(vrm);
    const seed=library.idle(new THREE.Vector3(),0), clip=SmplClip.from(seed.motion), driver=new VrmRetargeter(vrm,clip);
    root.position.copy(driver.rootAt(0));driver.apply(0);vrm.update(0);root.updateWorldMatrix(true,true);
    const bone=name=>vrm.humanoid.getNormalizedBoneNode(name);
    const pos=name=>bone(name).getWorldPosition(new THREE.Vector3());
    const initial=Object.fromEntries(['hips','head','rightUpperArm','rightHand','leftFoot','rightFoot'].map(name=>[name,pos(name).toArray()]));
    const obstacles=[{id:'seat',kind:'furniture',center:[0,.23,-.62],size:[2,.46,.64]}];
    const props=new InteractionProps([{id:'remote-test',label:'remote',center:[-.22,.468,.22],size:[.055,.036,.18]}]);scene.add(props.group);
    const body=new InteractionBody({vrm:()=>vrm,root,obstacles:()=>obstacles,props:()=>props});
    const run=async(operation,node)=>{
      let done=false,error=null,maxStep=0,footError=0;
      const feet=[pos('leftFoot'),pos('rightFoot')];
      let previous=pos('hips');
      const promise=body.execute('test-'+operation,operation,'fixture',node).then(()=>{done=true;},e=>{done=true;error=e.message;});
      for(let frame=0;frame<600&&!done;frame++){
        body.update(1/60);vrm.update(1/60);root.updateWorldMatrix(true,true);
        const current=pos('hips');maxStep=Math.max(maxStep,current.distanceTo(previous));previous=current;
        if(operation==='sit'||operation==='pickup')footError=Math.max(footError,pos('leftFoot').distanceTo(feet[0]),pos('rightFoot').distanceTo(feet[1]));
        await Promise.resolve();
      }
      if(!done)throw new Error('transition did not terminate');await promise;
      return {error,maxStep,footError,state:body.snapshot(),hips:pos('hips').toArray(),head:pos('head').toArray()};
    };
    const node={kind:'support',controlPoint:[0,.46,-.62],controlAxis:[1,0,0],size:[1.9,0,.62],sigma:.005,observedAt:Date.now()/1000,source:'measured-stereo-surface',frame:'control-world-y-up'};
    const results={initial,sit:await run('sit',node)};
    results.stand=await run('stand');
    results.lie=await run('lie',{...node,observedAt:Date.now()/1000});
    results.standAfterLie=await run('stand');
    results.pickup=await run('pickup',{...node,kind:'grasp',controlPoint:[-.22,.468,.22],size:[.055,.036,.18],observedAt:Date.now()/1000});
    results.release=await run('release');
    results.unsupported=await run('sit',{...node,controlPoint:[3,.46,-.62],observedAt:Date.now()/1000});
    results.cancel = await (async()=>{
      let error=null;
      const original=pos('hips');
      const pending=body.execute('cancel-test','sit','fixture',{...node,observedAt:Date.now()/1000}).catch(e=>{error=e.message;});
      for(let i=0;i<35;i++){body.update(1/60);await Promise.resolve();}
      const cancelled=body.cancel();let previous=pos('hips'),maxStep=0;
      for(let i=0;i<100;i++){body.update(1/60);vrm.update(1/60);await Promise.resolve();const current=pos('hips');maxStep=Math.max(maxStep,current.distanceTo(previous));previous=current;}
      await pending;await cancelled;
      return {error,maxStep,restoreError:pos('hips').distanceTo(original),state:body.snapshot()};
    })();
    for(const operation of ['sit','stand','lie','standAfterLie','pickup','release']){
      if(results[operation].error)throw new Error(operation+': '+results[operation].error);
      if(results[operation].maxStep>.02)throw new Error(operation+' hip discontinuity');
    }
    if(results.sit.footError>.002||results.pickup.footError>.002)throw new Error('feet slipped');
    if(!results.unsupported.error)throw new Error('unsupported seat accepted');
    if(!results.cancel.error||results.cancel.restoreError>.001||results.cancel.maxStep>.025||results.cancel.state.transitioning)throw new Error('cancellation left an invalid pose');
    window.bodyTest={THREE,vrm,root,scene,body,props,results};
    return results;
  });
}

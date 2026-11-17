async page => {
  return page.evaluate(async () => {
    const { THREE, vrm, root, scene } = window.bodyTest;
    const { InteractionBody } = await import('/src/interaction_body.ts');
    const { InteractionProps } = await import('/src/interaction_props.ts');
    const { capturePose, applyPose, point } = await import('/src/interaction_pose.ts');
    const original = capturePose(vrm);
    const props = new InteractionProps([
      { id:'food-fixture', label:'sandwich', kind:'food', portions:2, center:[-.22,.78,.22], size:[.16,.07,.14] },
      { id:'water-fixture', label:'cup', kind:'water', portions:3, center:[-.22,.94,.22], size:[.08,.12,.075] },
    ]); scene.add(props.group);
    const body = new InteractionBody({ vrm:()=>vrm, root, obstacles:()=>[], props:()=>props });
    const node = (position, size) => ({ kind:'grasp', controlPoint:position, size, controlAxis:[1,0,0], sigma:.005,
      observedAt:Date.now()/1000, source:'measured-stereo-surface', frame:'control-world-y-up' });
    const run = async (op, target, cancelAt) => {
      let done=false,error=null,maxStep=0,footError=0,maxJoint=0,contactFrames=0; const jumps=[];
      const feet=[point(vrm,'leftFoot'),point(vrm,'rightFoot')];
      let previous=point(vrm,'hips'), last=capturePose(vrm);
      const p=body.execute('life-'+op+'-'+Date.now(),op,'visual-fixture',target).then(()=>{done=true},e=>{done=true;error=e.message});
      let cancel;
      for(let frame=0;frame<2000&&!done;frame++){
        body.update(1/60);vrm.update(1/60);root.updateWorldMatrix(true,true);
        const current=point(vrm,'hips'),pose=capturePose(vrm);
        maxStep=Math.max(maxStep,current.distanceTo(previous));previous=current;
        const steps=pose.map((row,i)=>row.rotation.angleTo(last[i].rotation));
        const biggest=Math.max(...steps);if(biggest>.13)jumps.push({frame,angle:biggest,bone:pose[steps.indexOf(biggest)].node.name});
        maxJoint=Math.max(maxJoint,biggest);last=pose;
        footError=Math.max(footError,point(vrm,'leftFoot').distanceTo(feet[0]),point(vrm,'rightFoot').distanceTo(feet[1]));
        if(frame===cancelAt)cancel=body.cancel();
        if(frame>220&&frame%60===0)contactFrames++;
        await Promise.resolve();
      }
      if(!done)throw new Error(op+' never terminated');await p;if(cancel)await cancel;
      return {error,maxStep,maxJoint,footError,contactFrames,jumps,state:body.snapshot()};
    };
    const result={};
    for(const op of ['eat','drink']){
      const item=op==='eat'?[[-.22,.78,.22],[.16,.07,.14]]:[[-.22,.94,.22],[.08,.12,.075]];
      result[op]=await run(op,node(...item));
      if(result[op].error)throw new Error(op+': '+result[op].error);
      if(!result[op].state.consumption?.completed||result[op].state.heldObject)throw new Error(op+' missing consumption receipt');
      if(result[op].maxStep>.02||result[op].maxJoint>.13||result[op].footError>.004)throw new Error(op+' discontinuity '+JSON.stringify(result[op]));
      applyPose(original);
    }
    for(let style=1;style<3;style++){
      props.reset();
      result['drink-style-'+style]=await run('drink',node([-.22,.94,.22],[.08,.12,.075]));
      if(result['drink-style-'+style].error)throw new Error('variant '+style+': '+result['drink-style-'+style].error);
      applyPose(original);
    }
    props.reset();result.cancel=await run('eat',node([-.22,.78,.22],[.16,.07,.14]),260);
    if(!result.cancel.error||result.cancel.state.consumption?.completed)throw new Error('cancel rewarded consumption');
    if(result.cancel.state.heldObject)await run('release');
    result.unreachable=await run('drink',node([4,.94,.22],[.08,.12,.075]));
    if(!result.unreachable.error)throw new Error('unreachable drink accepted');
    body.reset(); props.dispose(); applyPose(original);
    return result;
  });
}

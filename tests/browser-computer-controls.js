async page => {
 const active = await page.evaluate(async()=>{
  const f=window.computerFixture;
  const pump=async(predicate,max=1800)=>{for(let i=0;i<max&&!predicate();i++){f.body.update(1/60);f.vrm.update(1/60);await Promise.resolve();}if(!predicate())throw new Error('fixture timeout');};
  let done=false,error;
  f.body.execute('stand-before-cancel','stand').then(()=>done=true,e=>{done=true;error=e.message});await pump(()=>done);if(error)throw new Error(error);
  f.node.observedAt=Date.now()/1000;f.node.seat.observedAt=Date.now()/1000;
  f.playPromise=f.body.execute('play-cancel','play_computer','fixture-keyboard',f.node,30).then(()=>({completed:true}),e=>({error:e.message}));
  await pump(()=>f.body.snapshot().computer?.active&&f.body.snapshot().computer.elapsed>4);
  f.renderer.render(f.scene,f.camera);
  const {point,palmPosition}=await import('/src/interaction_pose.ts');
  const gaps=['left','right'].map(side=>{
   const palm=palmPosition(f.vrm,side);
   const middle=point(f.vrm,side+'MiddleProximal'),wrist=point(f.vrm,side+'Hand');
   return {palm:palm.toArray(),fingerTilt:Math.abs(middle.y-wrist.y)/middle.distanceTo(wrist)};
  });
  return {state:f.body.snapshot(),hands:gaps};
 });
 await page.screenshot({path:'output/playwright/computer-room/playing.png'});
 const result = await page.evaluate(async()=>{
  const f=window.computerFixture;
  let done=false;const cancelled=f.body.cancel().then(()=>done=true);
  for(let i=0;i<200&&!done;i++){f.body.update(1/60);f.vrm.update(1/60);await Promise.resolve();}
  if(!done)throw new Error('cancel hung');await cancelled;
  const task=await f.playPromise,state=f.body.snapshot();
  if(!task.error||state.computer.active||state.computer.completed||state.transitioning||!state.supported)throw new Error('cancel did not restore supported sit');
  let standing=false,error;f.body.execute('stand-final','stand').then(()=>standing=true,e=>{standing=true;error=e.message});
  for(let i=0;i<250&&!standing;i++){f.body.update(1/60);f.vrm.update(1/60);await Promise.resolve();}
  if(error||!standing||f.body.snapshot().posture!=='stand')throw new Error(error??'stand failed');
  return {cancel:state,task,stand:f.body.snapshot()};
 });
 if(active.hands.some(hand=>hand.fingerTilt>.05))throw new Error('hands not level over keyboard');
 return {active,...result};
}

async (page) => page.evaluate(async () => {
  const {library,seed,THREE,cases}=window.motionTableTest;
  const {SmplClip}=await import('/src/retarget.ts');
  const {tableRotation,tableYaw}=await import('/src/motion_table_features.ts');
  const {presetSeed}=await import('/src/locomotion.ts');
  const result={repeated:[],long:[],regression:null};
  let currentSeed=seed;
  for(let index=0;index<12;index++){
    const input=SmplClip.from(currentSeed.motion),origin=input.rootAt(input.frames-1,new THREE.Vector3()),yaw=tableYaw(tableRotation(input,input.frames-1,0));
    const sign=index%2? -1:1;
    const path=Array.from({length:31},(_,i)=>{const t=i/30*.65;return new THREE.Vector3(sign*4*(1-Math.cos(t)),0,4*Math.sin(t)).applyAxisAngle(new THREE.Vector3(0,1,0),yaw).add(origin).toArray();});
    const begin=performance.now();
    try {
      const compiled=library.compileTable('chain-'+index,path,currentSeed,()=>null),clip=SmplClip.from(compiled.motion);
      const prefixUnchanged=currentSeed.motion.localRot.every((v,i)=>v===compiled.motion.localRot[i])&&currentSeed.motion.rootDelta.every((v,i)=>v===compiled.motion.rootDelta[i]);
      const stop=compiled.phases.find(p=>p.kind==='stop'),at=stop.start-.8;
      const nextSeed=presetSeed(clip,at,true,new THREE.Vector3());nextSeed.phase='walk';
      const roots=compiled.motion.rootDelta,k=currentSeed.motion.frames*3;
      const incoming=[roots[k]-roots[k-3],roots[k+2]-roots[k-1]],outgoing=k>=6?[roots[k-3]-roots[k-6],roots[k-1]-roots[k-4]]:[0,0];
      const rootAcceleration=Math.hypot(incoming[0]-outgoing[0],incoming[1]-outgoing[1])*400;
      result.repeated.push({index,ok:true,compileMs:performance.now()-begin,...compiled.table,quality:library.lastQuality,prefixUnchanged,
        startCount:compiled.phases.filter(p=>p.kind==='start').length,stopCount:compiled.phases.filter(p=>p.kind==='stop').length,
        seamSpeed:Math.hypot(...incoming)*20,rootAcceleration});
      currentSeed=nextSeed;
    }catch(error){result.repeated.push({index,ok:false,error:error.message});break;}
  }
  for(const length of [8,15,25]){
    const path=Array.from({length:81},(_,i)=>{const z=length*i/80;return [.7*Math.sin(z*.35),0,z];});
    const begin=performance.now();
    try{const compiled=library.compileTable('long-'+length,path,seed,()=>null);result.long.push({length,ok:true,compileMs:performance.now()-begin,...compiled.table,quality:library.lastQuality,frames:compiled.motion.frames});}
    catch(error){result.long.push({length,ok:false,compileMs:performance.now()-begin,error:error.message});}
  }
  const original=library.compile({id:'regression',duration:8,path:cases.straight,timing:'flexible',timeline:[{action:'forward',start:0,end:8,path:cases.straight,distance:3}]},seed,()=>null);
  result.regression={table:!!original.table,quality:library.lastQuality,end:original.motion.rootDelta.slice(-3),phases:original.phases.map(p=>p.kind)};
  let budgetRejected=false;try{library.compileTable('too-long',[[0,0,0],[0,0,50]],seed,()=>null);}catch(error){budgetRejected=error.message.includes('预算');}
  result.budgetRejected=budgetRejected;
  if(result.repeated.length!==12||result.repeated.some(row=>!row.ok||!row.prefixUnchanged||row.stopCount!==1||row.index>0&&(row.startCount!==0||row.seamSpeed<.1||row.rootAcceleration>4)))
    throw new Error(`连续续接回归 ${JSON.stringify(result.repeated)}`);
  if(result.long.some(row=>!row.ok||row.goalError>.12||row.pathMaxError>.3)||!budgetRejected)
    throw new Error(`长路线或预算回归 ${JSON.stringify(result.long)}`);
  if(result.regression.table||result.regression.quality.p95>.55) throw new Error('原生直线步态回归');
  return result;
})

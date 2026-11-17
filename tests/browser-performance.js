async (page) => {
  await page.waitForFunction(() => document.querySelector('#destination-select').options.length === 11);
  return await page.evaluate(async () => {
    const { SmplClip } = await import('/src/retarget.ts');
    const source = await fetch('/ardy/ardy_walk_long.json').then(response => response.json());
    const frames = Math.min(source.frames, 1200);
    function empty() { return {...source,frames:0,duration:0,localRot:[],rootDelta:[]}; }
    function extend(data, end) {
      const start=data.frames;
      data.localRot.push(...source.localRot.slice(start*81,end*81));
      data.rootDelta.push(...source.rootDelta.slice(start*3,end*3));
      data.frames=end; data.duration=end/20;
    }
    const full=empty();
    let started=performance.now();
    for(let end=40;end<=frames;end+=40) { extend(full,end); SmplClip.from(full); }
    const rebuildMs=performance.now()-started;
    const data=empty(); extend(data,40);
    const clip=SmplClip.from(data);
    started=performance.now();
    for(let end=80;end<=frames;end+=40) { extend(data,end); clip.appendFrames(); }
    const appendMs=performance.now()-started;
    const reference=SmplClip.from(data);
    let maxError=0;
    for(let i=0;i<reference.world.length;i++) maxError=Math.max(maxError,Math.abs(clip.world[i]-reference.world[i]));
    if(maxError>1e-6) throw new Error(`增量解码不一致 ${maxError}`);
    if(appendMs>=rebuildMs) throw new Error(`增量解码没有变快 ${appendMs}/${rebuildMs}`);
    return {frames:clip.frames,rebuildMs,appendMs,speedup:rebuildMs/appendMs,maxError,
      render:JSON.parse(document.querySelector('#performance-status').dataset.metrics)};
  });
}

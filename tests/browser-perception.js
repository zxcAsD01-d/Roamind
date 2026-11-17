async (page) => {
  const errors=[];
  const capture=error=>errors.push(error.message);
  page.on('pageerror',capture);
  try {
    await page.locator('#ego-toggle').check();
    await page.waitForFunction(()=>document.querySelector('#vision-status').textContent.includes('个物体'),null,{timeout:30000});
    await page.evaluate(async()=>{
      const context=new AudioContext();
      const audio=await fetch('/output/performance/asr-test.wav').then(response=>response.arrayBuffer());
      const buffer=await context.decodeAudioData(audio);
      const output=context.createMediaStreamDestination();
      const original=navigator.mediaDevices.getUserMedia;
      navigator.mediaDevices.getUserMedia=async()=>{
        await context.resume();const source=context.createBufferSource();
        source.buffer=buffer;source.connect(output);source.start();return output.stream;
      };
      window.motionAudioTest={context,original};
    });
    await page.locator('#mic-start').click();
    await page.waitForFunction(()=>document.querySelector('#speech-transcript').textContent.includes('研究'),null,{timeout:60000});
    const result={source:'synthetic-audio-worklet (no real microphone access)',
      vision:await page.locator('#vision-status').textContent(),
      audio:await page.locator('#audio-status').textContent(),
      transcript:await page.locator('#speech-transcript').textContent(),
      performance:JSON.parse(await page.locator('#performance-status').getAttribute('data-metrics')),errors};
    if(errors.length) throw new Error(errors.join('\n'));
    return result;
  } finally {
    if(await page.locator('#mic-stop').isEnabled()) await page.locator('#mic-stop').click();
    await page.evaluate(async()=>{
      const state=window.motionAudioTest;
      if(state){navigator.mediaDevices.getUserMedia=state.original;await state.context.close();delete window.motionAudioTest;}
    });
    page.off('pageerror',capture);
  }
}

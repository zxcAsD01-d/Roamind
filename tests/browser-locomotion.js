async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  return page.evaluate(async () => {
    const THREE = await import('/node_modules/.vite/deps/three.js');
    const { GLTFLoader } = await import('/node_modules/.vite/deps/three_addons_loaders_GLTFLoader__js.js');
    const { VRMLoaderPlugin, VRMUtils } = await import('/node_modules/.vite/deps/@pixiv_three-vrm.js');
    const { SmplClip, VrmRetargeter, measureVrmRest } = await import('/src/retarget.ts');
    const { LocomotionLibrary, presetSeed, PresetUnavailableError } = await import('/src/locomotion.ts');
    const { capturePresetBoundary, advancePresetBoundary } = await import('/src/motion_handoff.ts');
    const { HeadLookController } = await import('/src/head_look.ts');
    const loader = new GLTFLoader(); loader.register(parser => new VRMLoaderPlugin(parser));
    const gltf = await loader.loadAsync('/models/Wolferia.vrm');
    const vrm = gltf.userData.vrm;
    const group = new THREE.Group(); group.add(vrm.scene); vrm.scene.rotation.y = Math.PI;
    const bounds = new THREE.Box3().setFromObject(vrm.scene);
    const center = bounds.getCenter(new THREE.Vector3());
    vrm.scene.position.set(-center.x, -bounds.min.y, -center.z);
    measureVrmRest(vrm);
    const library = await LocomotionLibrary.load(vrm);
    const names = ['leftFoot', 'leftToes', 'rightFoot', 'rightToes'];
    const quantile = (values, fraction) => [...values].sort((a, b) => a-b)[Math.floor((values.length-1)*fraction)];
    const measure = (result, prefixFrames = 1) => {
      const clip = SmplClip.from(result.motion), driver = new VrmRetargeter(vrm, clip);
      const gaze = new HeadLookController(); gaze.explore();
      const feet = [], root = [], rotations = [];
      for (let frame = 0; frame < clip.frames; frame++) {
        gaze.restore();
        group.position.copy(driver.rootAt(frame)); driver.apply(frame); group.updateWorldMatrix(true, true);
        const rootForward = new THREE.Vector3(0,0,1).applyQuaternion(new THREE.Quaternion().fromArray(clip.local, frame*27*4));
        const phase = result.phases.find(item => item.start <= frame/20 && item.end > frame/20)?.kind;
        gaze.update(vrm, 1/20, Math.atan2(rootForward.x,rootForward.z), phase === 'walk', frame/20);
        feet.push(names.map(name => vrm.humanoid.getNormalizedBoneNode(name).getWorldPosition(new THREE.Vector3()).toArray()));
        root.push(group.position.toArray()); rotations.push(new THREE.Quaternion().fromArray(clip.local, frame*27*4));
      }
      const supportSpeed = [], rootSpeed = [], yawSteps = [];
      const floors = names.map((_, foot) => quantile(feet.map(frame => frame[foot][1]), .2));
      for (let frame = prefixFrames; frame < clip.frames; frame++) {
        rootSpeed.push(Math.hypot(root[frame][0]-root[frame-1][0], root[frame][2]-root[frame-1][2])*20);
        yawSteps.push(rotations[frame].angleTo(rotations[frame-1])*180/Math.PI);
        for (let foot = 0; foot < 4; foot++) {
          const a = feet[frame-1][foot], b = feet[frame][foot];
          if (Math.max(a[1], b[1]) < floors[foot]+.012 && Math.abs(b[1]-a[1])*20 < .08) {
            supportSpeed.push(Math.hypot(b[0]-a[0], b[2]-a[2])*20);
          }
        }
      }
      const footLift = names.map((_, foot) => Math.max(...feet.map(frame => frame[foot][1]))-Math.min(...feet.map(frame => frame[foot][1])));
      const heading = q => { const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(q); return Math.atan2(forward.x, forward.z)*180/Math.PI; };
      const final = root.at(-1);
      const head = new THREE.Quaternion().fromArray(clip.world, ((clip.frames-1)*clip.joints+6)*4);
      const headBearing = Math.atan2(Math.sin((heading(head)-heading(rotations.at(-1)))*Math.PI/180),
        Math.cos((heading(head)-heading(rotations.at(-1)))*Math.PI/180))*180/Math.PI;
      const actualHead = vrm.humanoid.getNormalizedBoneNode('head').getWorldQuaternion(new THREE.Quaternion())
        .multiply(measureVrmRest(vrm).get('head').clone().invert());
      const controlledHeadBearing = THREE.MathUtils.radToDeg(Math.atan2(
        Math.sin((heading(actualHead)-heading(rotations.at(-1)))*Math.PI/180),
        Math.cos((heading(actualHead)-heading(rotations.at(-1)))*Math.PI/180)));
      gaze.release();
      return { duration: result.seconds, end: final, heading: heading(rotations.at(-1)),
        headBearing, controlledHeadBearing,
        footLift, supportSamples: supportSpeed.length, supportMean: supportSpeed.reduce((a, b) => a+b, 0)/supportSpeed.length,
        supportP95: quantile(supportSpeed, .95), rootSpeedP95: quantile(rootSpeed, .95),
        maxYawFrameDegrees: Math.max(...yawSteps), phases: result.phases.map(phase => phase.kind),
        seamMaxJump: Math.max(...rootSpeed)/20, variants: result.phases.map(phase => phase.variant) };
    };
    const seed = library.idle(new THREE.Vector3(), 0);
    const compile = (action, path, heading) => library.compile({ id: action, duration: 4, path,
      timeline: [{ action, start: 0, end: 4, path, distance: path.length ? 1 : 0, heading,
        fixedHeading: !path.length }] }, seed, () => null);
    const walk = compile('walk', [[0, 0, 0], [0, 0, 3]]);
    const idle = [1, 2, 3].map(duration => {
      const motion = library.compile({ id: 'idle', duration, path: [],
        timeline: [{ action: 'observe', start: 0, end: duration, distance: 0 }] }, seed, () => null);
      const measured = measure(motion);
      if (measured.supportP95 > .16 || Math.hypot(measured.end[0], measured.end[2]) > .001)
        throw new Error(`待机滑步或持续漂移：${JSON.stringify(measured)}`);
      return measured;
    });
    const results = { idle, strideScale: library.strideScale, walk: measure(walk),
      shortWalk: measure(compile('walk', [[0, 0, 0], [0, 0, .65]])),
      turnLeft: measure(compile('turn_left', [], Math.PI/2)),
      turnRight: measure(compile('turn_right', [], -Math.PI/2)),
      turnAround: measure(compile('turn_left', [], Math.PI)),
      corner: measure(compile('walk', [[0, 0, 0], [0, 0, 1.5], [1.5, 0, 1.5]])) };
    const interrupted = presetSeed(SmplClip.from(walk.motion), 1.5, true, new THREE.Vector3());
    const after = library.compile({ id: 'interrupt-turn', duration: 2, path: [],
      timeline: [{ action: 'turn_left', start: 0, end: 2, distance: 0, heading: Math.PI/2, fixedHeading: true }] }, interrupted, () => null);
    results.interruption = measure(after, interrupted.motion.frames);
    const prefixUnchanged = after.motion.localRot.slice(0, interrupted.motion.localRot.length)
      .every((value, index) => value === interrupted.motion.localRot[index]);
    if (!prefixUnchanged) throw new Error('打断回写了已播放动作');
    if (Math.abs(results.walk.end[2]-3) > .03) throw new Error('根运动未到达目标');
    for (const key of ['turnLeft', 'turnRight', 'turnAround']) {
      if (Math.min(Math.max(...results[key].footLift.slice(0, 2)), Math.max(...results[key].footLift.slice(2))) < .025) throw new Error(`${key} 没有实际抬脚`);
      if (results[key].maxYawFrameDegrees > 12) throw new Error(`${key} 朝向跳变`);
      if (Math.abs(results[key].controlledHeadBearing) > 20) throw new Error(`${key} 结束后仍在侧看，无法稳定前进感知`);
      if (results[key].phases.includes('stop')) throw new Error(`${key} 转身后又拼接了站稳隔断`);
    }
    if (Math.abs(results.turnLeft.heading-90) > 1 || Math.abs(results.turnRight.heading+90) > 1) {
      throw new Error('转向动画未到达指定角度');
    }
    const flowing = (source) => {
      const end = source.motion.rootDelta.slice(-3);
      return library.compile({ id: 'continuous', duration: 6, path: [end, [end[0], end[1], end[2]+2.5]],
        timing: 'flexible', timeline: [{ action: 'forward', start: 0, end: 6, distance: 2.5, continuous: true }] }, source, () => null);
    };
    let stream = flowing(seed);
    const seams = [];
    for (let index = 0; index < 5; index++) {
      const brake = stream.phases.find(phase => phase.kind === 'stop');
      if (!brake) throw new Error('连续路线缺少断链停步备用片段');
      const spliceTime = brake.start-.9;
      const source = presetSeed(SmplClip.from(stream.motion), spliceTime, true, new THREE.Vector3());
      stream = flowing(source);
      if (!source.motion.localRot.every((value, offset) => stream.motion.localRot[offset] === value))
        throw new Error('提前续接回写了已播放前缀');
      if (stream.phases.some(phase => phase.kind === 'start')) throw new Error(`续接行走重复起步 ${JSON.stringify({index, spliceTime, brake, phases:stream.phases, tail:source.motion.rootDelta.slice(-9)})}`);
      const offset = source.motion.frames*3, roots = stream.motion.rootDelta;
      const speeds = [];
      for (let frame = -3; frame <= 4; frame++) {
        const at = offset+frame*3;
        speeds.push(Math.hypot(roots[at]-roots[at-3], roots[at+2]-roots[at-1])*20);
      }
      if (Math.min(...speeds) < .1) throw new Error(`续接缝出现停顿 ${JSON.stringify(speeds)}`);
      seams.push({ minimumSpeed: Math.min(...speeds), maximumSpeed: Math.max(...speeds), quality: library.lastQuality });
    }
    results.continuous = { seams, fallbackStop: measure(stream, stream.motion.frames-12) };
    const route = library.compile({ id: 'complete-route', duration: 10, timing: 'flexible',
      path: [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]],
      timeline: [{ action: 'forward', start: 0, end: 10, distance: 4, continuous: true }] }, seed, () => null);
    const stops = route.phases.filter(phase => phase.kind === 'stop');
    if (stops.length !== 1) throw new Error(`完整直线路线在路径点停步 ${JSON.stringify(route.phases)}`);
    const starts = route.phases.filter(phase => phase.kind === 'start');
    if (starts.length > 1) throw new Error('同一路线重复起步');
    const routeMetrics = measure(route);
    if (Math.hypot(routeMetrics.end[0], routeMetrics.end[2]-4) > .12) throw new Error(`整段路线终点偏移 ${JSON.stringify(routeMetrics.end)}`);
    const movingPhases = route.phases.filter(phase => phase.kind === 'walk');
    const seamSpeeds = movingPhases.slice(1).map(phase => {
      const index = Math.round(phase.start*20)*3, root = route.motion.rootDelta;
      return Math.hypot(root[index]-root[index-3], root[index+2]-root[index-1])*20;
    });
    if (seamSpeeds.some(speed => speed < .1)) throw new Error(`路径点步态速度归零 ${JSON.stringify(seamSpeeds)}`);
    results.completeRoute = { ...routeMetrics, stopCount: stops.length, startCount: starts.length, seamSpeeds };
    const chained = library.compile({id:'turn-walk-chain',duration:10,path:[],timing:'flexible',timeline:[
      {action:'turn_left',start:0,end:2,distance:0,heading:Math.PI/2,fixedHeading:true},
      {action:'forward',start:2,end:10,distance:2,path:[[0,0,0],[2,0,0]],continuous:true},
    ]},seed,()=>null);
    const firstWalk = chained.phases.findIndex(phase => phase.kind === 'walk' || phase.kind === 'start');
    if (firstWalk < 1 || chained.phases.slice(0,firstWalk).some(phase => phase.kind === 'idle' || phase.kind === 'stop'))
      throw new Error('转身接行走仍有待机隔断');
    results.chained = measure(chained);
    const turn = compile('turn_left', [], Math.PI/2);
    const tail = presetSeed(SmplClip.from(turn.motion), turn.motion.duration-.55, true, new THREE.Vector3(), true);
    tail.phase = 'turn';
    const endpoint = tail.motion.rootDelta.slice(-3);
    const afterTurn = library.compile({id:'turn-prefetch',duration:7,timing:'flexible',
      path:[endpoint,[endpoint[0]+2,endpoint[1],endpoint[2]]],
      timeline:[{action:'forward',start:0,end:7,distance:2,continuous:true}]},tail,()=>null);
    if (!tail.motion.localRot.every((value,index)=>value===afterTurn.motion.localRot[index])
        || !tail.motion.rootDelta.every((value,index)=>value===afterTurn.motion.rootDelta[index]))
      throw new Error('转身预取改写了仍在播放的原生收尾');
    if (afterTurn.phases[0].kind === 'stop' || afterTurn.phases[0].kind === 'idle')
      throw new Error('转身预取后又插入站稳隔断');
    results.turnPrefetch = {retainedSeconds:tail.motion.duration-tail.time,phases:afterTurn.phases.map(phase=>phase.kind)};
    const turnClip = SmplClip.from(turn.motion), capturedTime = (turnClip.frames-1)/20-.55;
    const boundary = capturePresetBoundary(turnClip, capturedTime, true, new THREE.Vector3(), turn.phases, true);
    if (boundary.seed.phase !== 'turn' || Math.abs(boundary.heading-Math.PI/2) > .02)
      throw new Error('转身前瞻未使用尾帧阶段及朝向');
    const advanced = advancePresetBoundary(boundary, turnClip, capturedTime+.15);
    if (Math.abs(advanced.time-boundary.seed.time-.15) > 1e-6) throw new Error('异步规划未补偿已播放时间');
    if (advanced.motion !== boundary.seed.motion) throw new Error('异步规划重新抓取了不一致的前缀');
    let rejected = false;
    try { advancePresetBoundary(boundary, turnClip, capturedTime+2); } catch { rejected = true; }
    if (!rejected) throw new Error('已过期的规划前缀仍被接受');
    results.boundary = { phase: boundary.seed.phase, heading: boundary.heading, latencySeconds: .15, expiredRejected: rejected };
    const stoppedClip = SmplClip.from(walk.motion), stoppedAt = (stoppedClip.frames-1)/stoppedClip.fps;
    const stoppedBoundary = capturePresetBoundary(stoppedClip, stoppedAt, false, new THREE.Vector3(), walk.phases);
    const stoppedSeed = advancePresetBoundary(stoppedBoundary, stoppedClip, stoppedAt);
    if (Math.abs(stoppedSeed.time-(stoppedSeed.motion.frames-1)/stoppedSeed.motion.fps) > 1e-6)
      throw new Error('已停稳的尾帧被误判为过期');
    const endingBoundary = capturePresetBoundary(stoppedClip, stoppedAt-.1, true, new THREE.Vector3(), walk.phases, true);
    const endingSeed = advancePresetBoundary(endingBoundary, stoppedClip, stoppedAt+.03);
    if (endingSeed.time > (endingSeed.motion.frames-1)/endingSeed.motion.fps+.001)
      throw new Error('完整尾段自然结束后规划失效');
    for (let frame = 0; frame < stoppedClip.frames; frame++) {
      const seed = presetSeed(stoppedClip, frame/stoppedClip.fps, false, new THREE.Vector3());
      if (seed.time > (seed.motion.frames-1)/seed.motion.fps+.001)
        throw new Error('浮点取整使静止前缀失效');
    }
    results.boundary.stoppedFrameChecks = stoppedClip.frames;
    const movingClip = SmplClip.from(walk.motion), movingAt = 1.5;
    const movingBoundary = capturePresetBoundary(movingClip, movingAt, true, new THREE.Vector3(), walk.phases, false, true);
    const retained = (movingBoundary.seed.motion.frames-1)/20-movingBoundary.seed.time;
    if (retained < .6 || retained > .9) throw new Error('行走续接未保留网络延迟预算：'+retained);
    const delayedSeed = advancePresetBoundary(movingBoundary, movingClip, movingAt+.45);
    const delayedEnd = delayedSeed.motion.rootDelta.slice(-3);
    const delayed = library.compile({id:'delayed-continuation',duration:6,timing:'flexible',
      path:[delayedEnd,[delayedEnd[0],delayedEnd[1],delayedEnd[2]+2.5]],
      timeline:[{action:'forward',start:0,end:6,distance:2.5,continuous:true}]},delayedSeed,()=>null);
    if (!delayedSeed.motion.localRot.every((value,index)=>value===delayed.motion.localRot[index])
      || !delayedSeed.motion.rootDelta.every((value,index)=>value===delayed.motion.rootDelta[index]))
      throw new Error('延迟续接改写原动作前缀');
    if (delayed.phases.some(phase=>phase.kind==='start')) throw new Error('延迟续接重复起步');
    const brake = walk.phases.find(phase=>phase.kind==='stop');
    const lateAt = brake.start-.35;
    const lateBoundary = capturePresetBoundary(movingClip,lateAt,true,new THREE.Vector3(),walk.phases,false,true);
    const cutAt = lateAt+(lateBoundary.seed.motion.frames-1)/20-lateBoundary.seed.time;
    if (cutAt >= brake.start) throw new Error('行走前缀包含收步阶段');
    let expired = false;
    try {advancePresetBoundary(movingBoundary,movingClip,movingAt+1.2);} catch {expired=true;}
    if (!expired) throw new Error('超过保留时长的续接仍被接受');
    results.delayedContinuation={retainedSeconds:retained,latencySeconds:.45,
      phase:movingBoundary.seed.phase,lateCutAt:cutAt,brakeAt:brake.start,expiredRejected:expired};
    const savedAssets = library.assets;
    try {
      library.assets = savedAssets.filter(asset => asset.motion.kind !== 'turn');
      let unavailable = false;
      try { compile('turn_left', [], Math.PI/2); }
      catch (error) {
        if (!(error instanceof PresetUnavailableError)) throw error;
        unavailable = true;
      }
      if (!unavailable) throw new Error('缺少转身能力时没有返回可恢复的动作不可用状态');
      results.missingTurnRecoverable = unavailable;
    } finally { library.assets = savedAssets; }
    VRMUtils.deepDispose(vrm.scene);
    return { ...results, prefixUnchanged };
  });
}

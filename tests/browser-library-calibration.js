async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  return page.evaluate(async () => {
    const THREE = await import('/node_modules/.vite/deps/three.js');
    const { GLTFLoader } = await import('/node_modules/.vite/deps/three_addons_loaders_GLTFLoader__js.js');
    const { VRMLoaderPlugin, VRMUtils } = await import('/node_modules/.vite/deps/@pixiv_three-vrm.js');
    const { SmplClip, VrmRetargeter, measureVrmRest } = await import('/src/retarget.ts');
    const { LocomotionLibrary } = await import('/src/locomotion.ts');
    const quantile = (values, p) => [...values].sort((a, b) => a-b)[Math.floor((values.length-1)*p)];
    const report = [];
    for (const file of ['Wolferia.vrm', 'kyoko.vrm', 'NEKONA.01.vrm']) {
      const loader = new GLTFLoader(); loader.register(parser => new VRMLoaderPlugin(parser));
      const gltf = await loader.loadAsync(`/models/${file}`), vrm = gltf.userData.vrm;
      const group = new THREE.Group(); group.add(vrm.scene); vrm.scene.rotation.y = Math.PI;
      const bounds = new THREE.Box3().setFromObject(vrm.scene), center = bounds.getCenter(new THREE.Vector3());
      vrm.scene.position.set(-center.x, -bounds.min.y, -center.z); measureVrmRest(vrm);
      const library = await LocomotionLibrary.load(vrm);
      const measure = result => {
        const clip = SmplClip.from(result.motion), driver = new VrmRetargeter(vrm, clip), samples = [];
        for (let frame = 0; frame < clip.frames; frame++) {
          group.position.copy(driver.rootAt(frame)); driver.apply(frame); group.updateWorldMatrix(true, true);
          samples.push(['leftFoot', 'leftToes', 'rightFoot', 'rightToes'].map(name =>
            vrm.humanoid.getNormalizedBoneNode(name).getWorldPosition(new THREE.Vector3()).toArray()));
        }
        const floors = [0, 1, 2, 3].map(foot => quantile(samples.map(frame => frame[foot][1]), .2)), speeds = [];
        for (let frame = 1; frame < samples.length; frame++) for (let foot = 0; foot < 4; foot++) {
          const a = samples[frame-1][foot], b = samples[frame][foot];
          if (Math.max(a[1], b[1]) < floors[foot]+.012 && Math.abs(b[1]-a[1])*20 < .08)
            speeds.push(Math.hypot(b[0]-a[0], b[2]-a[2])*20);
        }
        const q = new THREE.Quaternion().fromArray(clip.local, (clip.frames-1)*27*4);
        const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
        return { seconds: result.seconds, end: driver.rootAt(clip.frames-1).toArray(),
          heading: Math.atan2(forward.x, forward.z)*180/Math.PI,
          supportSamples: speeds.length, supportMean: speeds.reduce((a, b) => a+b, 0)/speeds.length,
          supportP95: quantile(speeds, .95), variants: result.phases.flatMap(phase => phase.variant ? [phase.variant] : []) };
      };
      const cases = [];
      for (let index = 0; index < 9; index++) {
        const distance = [.65, 1.5, 3][Math.floor(index/3)], seed = library.idle(new THREE.Vector3(), 0);
        const started = performance.now();
        const result = library.compile({ id: `audit-${index}`, duration: 12, path: [[0, 0, 0], [0, 0, distance]],
          timeline: [{ action: 'walk', start: 0, end: 12, distance }] }, seed, () => null);
        const compileMs = performance.now()-started;
        const measured = measure(result);
        const error = Math.hypot(measured.end[0], measured.end[2]-distance);
        if (error > .25 || measured.supportP95 > .55 || measured.supportMean > .16 || !measured.supportSamples)
          throw new Error(`${file} 行走校准未通过：${JSON.stringify({ distance, error, ...measured })}`);
        cases.push({ distance, ...measured, error, compileMs, quality: library.lastQuality });
      }
      const walkVariants = new Set(cases.flatMap(value => value.variants.filter(name => name.startsWith('walk'))));
      if (walkVariants.size < 3) throw new Error(`${file} 未使用三套行走变体`);
      const turns = [];
      for (let index = 0; index < 6; index++) {
        const heading = index%2 ? -Math.PI/2 : Math.PI/2;
        turns.push(measure(library.compile({ id: `turn-${index}`, duration: 8, path: [],
          timeline: [{ action: 'turn_left', start: 0, end: 8, distance: 0, heading }] },
        library.idle(new THREE.Vector3(), 0), () => null)));
      }
      if (turns.some(value => Math.abs(Math.abs(value.heading)-90) > 1 || value.supportP95 > .55))
        throw new Error(`${file} 转向质量未通过`);
      report.push({ file, calibration: library.calibration, cases, turns });
      VRMUtils.deepDispose(vrm.scene);
    }
    return report;
  });
}

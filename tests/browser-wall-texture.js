async page => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('details').evaluateAll(items => items.forEach(item => { item.open = true; }));
  await page.locator('#ambient-idle').uncheck();
  await page.locator('#ego-toggle').uncheck();
  return page.evaluate(async () => {
    const THREE = await import('/node_modules/.vite/deps/three.js');
    const { HouseScene } = await import('/src/house.ts');
    const { StereoCapture } = await import('/src/stereo.ts');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x9ca9aa);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x60605c, 2.4));
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(640, 360);
    const camera = new THREE.PerspectiveCamera(78, 640/360, .06, 60);
    const world = { id: 'wall-evaluation-only', rooms: [], objects: [
      { id: 'wall', kind: 'wall', material: 'wall', center: [0, 1.5, -.08], size: [24, 3, .16] }],
      bounds: [-12, -12, 12, 12], spawn: [0, .92, 2], agent: { radius: .32, height: 1.7, skin: .08 } };
    const house = new HouseScene(world, scene);
    house.setCutaway(false);
    const wall = house.group.children.find(child => child.isInstancedMesh);
    const capture = new StereoCapture();
    const post = async (path, body) => {
      for (let attempt = 0; attempt < 4; attempt++) {
        const response = await fetch('http://127.0.0.1:8200'+path, { method: 'POST',
          headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (response.status === 429 && attempt < 3) {
          await new Promise(resolve => setTimeout(resolve, 200));
          continue;
        }
        const result = await response.json();
        if (!response.ok) throw new Error(path+' HTTP '+response.status+' '+JSON.stringify(result));
        return result;
      }
      throw new Error('Stereo service retry exhausted');
    };
    const samples = [];
    let preview = '';
    try {
      for (const distance of [1.5, 3, 6]) {
        const clientId = 'wall-evaluation-'+crypto.randomUUID();
        const frames = [];
        try {
          for (let frame = 0; frame < 7; frame++) {
            camera.position.set(frame*.06, 1.5, distance);
            camera.lookAt(frame*.06, 1.5, 0);
            camera.updateMatrixWorld(true);
            const data = capture.capture(scene, renderer, camera);
            renderer.setRenderTarget(null);
            const result = await post('/api/stereo', { ...data, clientId });
            const depths = [];
            let valid = 0, accurate = 0;
            for (let row = 18; row < 27; row++) for (let column = 20; column < 60; column++) {
              const depth = result.depthGrid.meters[row*80+column];
              if (depth > 0) { valid++; depths.push(Math.abs(depth-distance)); }
              if (depth > 0 && Math.abs(depth-distance)/distance < .15) accurate++;
            }
            depths.sort((a, b) => a-b);
            frames.push({ validRatio: result.validRatio, wallCoverage: valid/360, accurateCoverage: accurate/360,
              medianErrorMeters: depths.length ? depths[Math.floor(depths.length/2)] : null,
              poseStatus: result.pose.status, inliers: result.pose.inliers, inferenceMs: result.inferenceMs });
            if (distance === 3 && frame === 0) preview = data.left;
          }
        } finally { await post('/api/stereo/release', { clientId }); }
        samples.push({ distance, frames });
      }
      renderer.render(scene, camera);
      return { wallTextured: !!wall.material.map, textureSize: wall.material.map?.image.width ?? 0,
        drawCalls: renderer.info.render.calls, samples, preview };
    } finally {
      scene.traverse(object => {
        object.geometry?.dispose();
        if (object.material) for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
          material.map?.dispose(); material.dispose();
        }
      });
      renderer.dispose();
    }
  });
}

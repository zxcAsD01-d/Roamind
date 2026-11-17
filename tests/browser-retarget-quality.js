async (page) => {
  return page.evaluate(async () => {
    const THREE = await import('/node_modules/.vite/deps/three.js');
    const { GLTFLoader } = await import('/node_modules/.vite/deps/three_addons_loaders_GLTFLoader__js.js');
    const { VRMLoaderPlugin, VRMUtils } = await import('/node_modules/.vite/deps/@pixiv_three-vrm.js');
    const { SmplClip, VrmRetargeter, measureVrmRest } = await import('/src/retarget.ts');
    const loader = new GLTFLoader();
    loader.register(parser => new VRMLoaderPlugin(parser));
    const motion = await fetch('/output/quality/free-before.json').then(response => response.json());
    const clip = SmplClip.from(motion);
    const results = [];
    for (const file of ['Wolferia.vrm', 'kyoko.vrm', 'NEKONA.01.vrm']) {
      const gltf = await loader.loadAsync(`/models/${file}`);
      const vrm = gltf.userData.vrm;
      if (!vrm) throw new Error(`${file}: VRM missing`);
      vrm.scene.rotation.y = Math.PI;
      const bounds = new THREE.Box3().setFromObject(vrm.scene);
      const center = bounds.getCenter(new THREE.Vector3());
      vrm.scene.position.set(-center.x, -bounds.min.y, -center.z);
      const group = new THREE.Group();
      group.add(vrm.scene);
      measureVrmRest(vrm);
      const names = ['hips', 'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes',
        'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes'];
      const positions = () => Object.fromEntries(names.map(name => {
        const bone = vrm.humanoid.getNormalizedBoneNode(name);
        return [name, bone?.getWorldPosition(new THREE.Vector3()).toArray() ?? null];
      }));
      const rest = positions();
      const retargeter = new VrmRetargeter(vrm, clip);
      const samples = [];
      const localOnly = [];
      for (let frame = 0; frame < clip.frames; frame++) {
        group.position.copy(retargeter.rootAt(frame));
        retargeter.apply(frame);
        vrm.update(0);
        group.updateMatrixWorld(true);
        samples.push(positions());
        group.position.set(0, 0, 0);
        group.updateMatrixWorld(true);
        localOnly.push(positions());
      }
      results.push({ file, rest, samples, localOnly });
      VRMUtils.deepDispose(vrm.scene);
    }
    return results;
  });
}

async page => {
  return page.evaluate(async () => {
    const params = new URL(location.href).searchParams;
    const directory = params.get('bridgeDirectory') ?? '/output/supermimic/avatar-bridge';
    const reports = await (await fetch(`${directory}/report.json`)).json();
    const cases = [];
    for (const entry of reports) {
      const source = await (await fetch(`${directory}/${entry.id}-source.json`)).json();
      const candidate = entry.tracking.completed ? await (await fetch(`${directory}/${entry.id}-candidate.json`)).json() : null;
      cases.push({ name: entry.name, source, candidate, accepted: entry.accepted,
        expected: entry.qualitySpace === 'measured avatar geometry' ? entry.after : null });
    }
    const THREE = await import('/node_modules/.vite/deps/three.js');
    const { GLTFLoader } = await import('/node_modules/.vite/deps/three_addons_loaders_GLTFLoader__js.js');
    const { VRMLoaderPlugin } = await import('/node_modules/.vite/deps/@pixiv_three-vrm.js');
    const { SmplClip, VrmRetargeter, measureVrmRest } = await import('/src/retarget.ts');
    const { measureSupport } = await import('/src/motion_quality.ts');
    const loader = new GLTFLoader(); loader.register(parser => new VRMLoaderPlugin(parser));
    const avatar = params.get('bridgeAvatar') ?? 'Wolferia';
    const vrm = (await loader.loadAsync(`/models/${encodeURIComponent(avatar)}.vrm`)).userData.vrm;
    const group = new THREE.Group(); group.add(vrm.scene); vrm.scene.rotation.y = Math.PI;
    const bounds = new THREE.Box3().setFromObject(vrm.scene), center = bounds.getCenter(new THREE.Vector3());
    vrm.scene.position.set(-center.x, -bounds.min.y, -center.z); measureVrmRest(vrm);
    const feet = ['leftFoot', 'leftToes', 'rightFoot', 'rightToes'].map(name => vrm.humanoid.getNormalizedBoneNode(name));
    const bones = Object.keys(vrm.humanoid.humanBones).map(name => vrm.humanoid.getNormalizedBoneNode(name)).filter(Boolean);
    const saved = bones.map(bone => bone.quaternion.clone());
    const positions = Object.fromEntries(Object.keys(vrm.humanoid.humanBones)
      .filter(name => cases.some(entry => entry.source.bones.includes(name)))
      .map(name => [name, vrm.humanoid.getNormalizedBoneNode(name).getWorldPosition(new THREE.Vector3()).toArray()]));
    function calibration(motion) {
      const clip = SmplClip.from(motion), driver = new VrmRetargeter(vrm, clip), positions = [];
      try {
        for (let frame = 0; frame < clip.frames; frame++) {
          driver.apply(frame);
          positions.push(feet.map(foot => foot.getWorldPosition(new THREE.Vector3())));
        }
      } finally { bones.forEach((bone, i) => bone.quaternion.copy(saved[i])); group.updateWorldMatrix(true, true); }
      const floors = feet.map((_, i) => positions.map(frame => frame[i].y).sort((a,b) => a-b)[Math.floor(positions.length*.3)]);
      const ratios = [];
      for (let frame = 1; frame < positions.length; frame++) {
        const dr = clip.rootAt(frame, new THREE.Vector3()).sub(clip.rootAt(frame-1, new THREE.Vector3())); dr.y=0;
        if (dr.lengthSq() < .0001) continue;
        feet.forEach((_, foot) => {
          if (Math.max(positions[frame][foot].y, positions[frame-1][foot].y) > floors[foot]+.018) return;
          const df=positions[frame][foot].clone().sub(positions[frame-1][foot]); df.y=0;
          const ratio=-df.dot(dr)/dr.lengthSq(); if(ratio>.25&&ratio<1.5)ratios.push(ratio);
        });
      }
      ratios.sort((a,b)=>a-b);
      return ratios.length>=12 ? ratios[Math.floor(ratios.length/2)] : null;
    }
    const comparisons = cases.map(entry => {
      if (entry.source.fps !== 20) throw new Error('Existing VRM support meter requires 20 fps comparison clips');
      const scale = calibration(entry.source) ?? .75;
      const result = { name: entry.name, scale, accepted: entry.accepted };
      for (const label of ['source','candidate']) {
        const original = entry[label];
        if (!original) continue;
        const motion = {...original, rootDelta: original.rootDelta.map(value=>value*scale)};
        result[label] = measureSupport(vrm,motion,[],0,true);
      }
      result.improved = Boolean(result.candidate && result.candidate.mean<result.source.mean*.95 && result.candidate.p95<=result.source.p95+.005);
      if (entry.expected && result.candidate && Math.abs(result.candidate.mean-entry.expected.supportMean)>1e-6)
        throw new Error(`${entry.name}: measured avatar FK differs from original VRM retargeting`);
      if (entry.accepted && !result.improved) throw new Error(`${entry.name}: accepted correction regressed on real VRM`);
      return result;
    });
    return { profile: {version: 1, avatar, positions,
      rootScales: Object.fromEntries(comparisons.map(entry => [entry.name, entry.scale]))}, comparisons };
  });
}

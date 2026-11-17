async page => page.evaluate(async () => {
  const { MotionSearch, boundaryFeatures, featureCost } = await import('/src/motion_search.ts');
  const response = await fetch('/locomotion/bank.json');
  if (!response.ok) throw new Error('bank HTTP '+response.status);
  const bank = await response.json();
  let queries = 0;
  const started = performance.now();
  for (const clip of bank.clips) {
    const index = new MotionSearch(clip);
    for (const row of clip.search.entries) {
      const query = boundaryFeatures(clip, row[0]);
      if (query.length !== 46 || !query.every(Number.isFinite)) throw new Error('Invalid boundary features');
      if (Math.max(...query.slice(2, 28).map((v, i) => Math.abs(v-row[i+3]))) > .00003)
        throw new Error('Python/TS support and quaternion feature mismatch: '+clip.name);
      const match = index.shortlist(query, row[0], row[0], 1, 1)[0];
      if (!match || match.frame !== row[0] || !Number.isFinite(match.cost)) throw new Error('Missing narrow-window match');
      queries++;
    }
    if (clip.frames > 16) {
      const match = index.shortlist(boundaryFeatures(clip, 15), 15, 15, 1, 1)[0];
      if (match?.frame !== 15) throw new Error('Odd entry between offline samples was lost');
    }
    const q = boundaryFeatures(clip, 2), equivalent = [...q];
    for (let i = 4; i < 28; i++) equivalent[i] *= -1;
    if (Math.abs(featureCost(q, q, 1, 1)-featureCost(q, equivalent, 1, 1)) > 1e-9)
      throw new Error('Quaternion sign changed score');
  }
  const clip = bank.clips[0];
  for (const entries of [[], [[-1, ...new Array(46).fill(0)]], [[0, 1]]]) {
    let rejected = false;
    try { new MotionSearch({ ...clip, search: { version: 1, entries } }); } catch { rejected = true; }
    if (!rejected) throw new Error('Malformed index accepted');
  }
  return { clips: bank.clips.length, queries, elapsedMs: performance.now()-started, passed: true };
})

// Research adapter: imports original HAST-SEC modules; does not patch the vendor tree.
// Upstream license and third-party notices remain in external/HAST-SEC/.
import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {Quaternion, Vector3} from 'three';
import {beatLayers} from '../vendor/HAST-SEC/motion/beat_layers.mjs';
import {beatRuntime} from '../vendor/HAST-SEC/training/beat_runtime.mjs';
import {beatRuntimeV9} from '../vendor/HAST-SEC/training/beat_runtime_v9.mjs';
import {outputSmoothing} from '../vendor/HAST-SEC/motion/v9/output_smoothing.mjs';
import {rotationExp, rotationLog} from '../vendor/HAST-SEC/retarget/rotation_math.mjs';

const vendor = new URL('../external/HAST-SEC/', import.meta.url);
const readJson = async path => JSON.parse(await readFile(new URL(path, vendor), 'utf8'));
const dt = 1/20, timings = [];
const finitePoint = point => point.p.toArray().every(Number.isFinite)
  && point.q.every(q => q.toArray().every(Number.isFinite) && Math.abs(q.lengthSq()-1) < 1e-5);
const quaternions = local => local.map(v => rotationExp(new Vector3(...v)));
const rotvecs = qs => qs.map(q => rotationLog(q).toArray());
const clonePoint = point => ({q: point.q.map(q => q.clone()), p: point.p.clone()});

async function readFloats(path) {
  const bytes = await readFile(new URL(path, vendor));
  assert.equal(bytes.byteLength%4, 0);
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset+bytes.byteLength));
}

async function resources() {
  const meta = await readJson('models/body/model.json');
  meta.prototypes = await readJson('models/body/prototypes.json');
  return {meta, weights: await readFloats('models/body/weights.bin'),
    codes: await readFloats('models/body/codes.bin'), settings: await readJson('configs/v9_control.json')};
}

function timeCall(method, sample) {
  const begin = performance.now(), value = sample();
  timings.push({method, ms: performance.now()-begin});
  return value;
}

function residualFrames(test, resource) {
  const count = test.local.length, dim = resource.meta.dim;
  // Pad 27 independent channels to the upstream 53-channel contract. This does
  // not map HAST skeletal rest poses into SMPL or change the VRM retargeter.
  const meta = {...resource.meta, frames: count, excluded_prototypes: [],
    bones: [...test.bones, ...Array.from({length: 26}, (_, i) => 'unused'+i)]};
  const codes = new Float32Array(meta.codes*count*dim);
  for (const [token, frames] of [test.oldLocal, test.local].entries()) {
    frames.forEach((frame, i) => codes.set(frame.flat(), (token*count+i)*dim));
  }
  let forced = 0;
  const runtime = beatRuntimeV9(meta, resource.weights, codes, {policy: token => token === forced ? 1 : 0});
  runtime.sample(0);
  for (let i = 1; i < count-1; i++) runtime.sample(dt);
  forced = 1;
  const first = timeCall('hastResidual', () => runtime.sample(dt+1e-9));
  assert.equal(first.token, 1, 'forced transition must select incoming clip');
  const rows = [first];
  for (let i = 1; i < count; i++) {
    rows.push(timeCall('hastResidual', () => runtime.sample(i === count-1 ? dt-2e-9 : dt)));
  }
  return {local: rows.map(row => Array.from({length: 27}, (_, i) => row.values.slice(i*3, i*3+3))),
    root: test.root, transitionSeconds: first.transition_duration};
}

function layerFrames(test) {
  const source = new Float32Array(test.local.flatMap(frame => quaternions(frame).flatMap(q => q.toArray())));
  const base = quaternions(test.oldLocal.at(-1)).map(q => q.toArray());
  const runtime = beatLayers(test.bones, new Map([['test', source]]));
  runtime.add({key: 'test', mask: test.bones.map((_, i) => i),
    skills: [{source: 'test', start: 0, end: test.local.length, duration: (test.local.length-1)*dt}]},
  {hold: 0, cycles: 1}, base);
  const rows = test.local.map((_, i) => rotvecs(timeCall('hastLayers', () => runtime.sample(base, i ? dt : 0))
    .map(value => new Quaternion(...value))));
  return {local: rows, root: test.root, fadeSeconds: runtime.state().tracks[0]?.fade_seconds};
}

function dampedFrames(test, resource, preserveRoot = false) {
  const runtime = outputSmoothing(test.bones, resource.settings);
  const point = (local, root) => ({q: quaternions(local), p: new Vector3(...root)});
  const history = [];
  for (let i = 0; i < test.oldLocal.length; i++) {
    const incoming = point(test.oldLocal[i], test.oldRoot[i]);
    const map = preserveRoot ? p => ({q: p.q, p: incoming.p.clone()}) : p => p;
    history.push(runtime.step(incoming, dt, null, map, finitePoint));
  }
  const method = preserveRoot ? 'hastDampedPreserveRoot' : 'hastDamped';
  const frames = test.local.map((local, i) => {
    const incoming = point(local, test.root[i]);
    const map = preserveRoot ? p => ({q: p.q, p: incoming.p.clone()}) : p => p;
    return timeCall(method, () => runtime.step(incoming, dt, null, map, finitePoint));
  });
  return {local: frames.map(frame => rotvecs(frame.q)), root: frames.map(frame => frame.p.toArray()),
    oldLocal: history.map(frame => rotvecs(frame.q)), oldRoot: history.map(frame => frame.p.toArray())};
}

async function nativeSources() {
  const manifests = await Promise.all([0, 1, 2, 3].map(i => readJson('models/skills/sources-'+i+'.json')));
  const descriptors = manifests.flat(), sources = new Map();
  for (const source of descriptors) {
    const bytes = await readFile(new URL('models/trajectories/'+source.file, vendor));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), source.sha256);
    assert.equal(bytes.byteLength, source.frames*53*16);
    sources.set(source.id, new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset+bytes.byteLength)));
  }
  return {descriptors, sources};
}

function nativeLayerCase(entry, skill, base, bones, sources) {
  const layers = beatLayers(bones, sources), durations = [], frames = [];
  layers.add({...entry, skills: [skill]}, {cycles: 4, hold: 0}, base);
  const total = skill.duration*4, count = Math.ceil(total/dt)+1;
  let unaffectedMax = 0, maxStep = 0, cycleBoundaryMax = 0, peakWeight = 0, last = base;
  for (let i = 0; i < count; i++) {
    const begin = performance.now(), value = layers.sample(base, i ? dt : 0);
    durations.push(performance.now()-begin);
    for (let j = 0; j < bones.length; j++) {
      const q = new Quaternion(...value[j]);
      assert.ok(q.toArray().every(Number.isFinite) && Math.abs(q.lengthSq()-1) < 1e-5);
      if (!entry.mask.includes(j)) unaffectedMax = Math.max(unaffectedMax, q.angleTo(new Quaternion(...base[j])));
      maxStep = Math.max(maxStep, q.angleTo(new Quaternion(...last[j])));
      if (i && Math.floor(i*dt/skill.duration) !== Math.floor((i-1)*dt/skill.duration)) {
        cycleBoundaryMax = Math.max(cycleBoundaryMax, q.angleTo(new Quaternion(...last[j])));
      }
    }
    peakWeight = Math.max(peakWeight, ...layers.state().weights);
    frames.push(value); last = value;
  }
  assert.ok(unaffectedMax < 1e-6, 'layer must preserve unmasked joints');
  return {layers, frames, durations, result: {key: entry.key, source: skill.source, start: skill.start,
    maxStepRad: maxStep, cycleBoundaryMaxRad: cycleBoundaryMax, peakWeight, unaffectedMaxRad: unaffectedMax}};
}

function interruptNative(entry, base, bones, sources) {
  const layers = beatLayers(bones, sources), first = entry.skills[0], second = entry.skills.at(-1);
  layers.add({...entry, skills: [first]}, {cycles: 4, hold: 0}, base);
  let current;
  for (let i = 0; i < Math.ceil(first.duration/dt/2); i++) current = layers.sample(base, dt);
  layers.add({...entry, skills: [second]}, {cycles: 2, hold: 0}, current);
  assert.deepEqual(layers.sample(base, 0), current, 'switch must preserve the displayed pose at t=0');
  const next = layers.sample(base, dt);
  layers.clear();
  assert.deepEqual(layers.sample(base, 0), next, 'clear must begin from the displayed weight');
  return Math.max(...next.map((q, i) => new Quaternion(...q).angleTo(new Quaternion(...current[i]))));
}

async function nativeLibrary(resource) {
  const manifest = await readJson('models/skills/manifest.json');
  const extra = await readJson('models/control/skills.json');
  assert.deepEqual(manifest.bones, extra.bones);
  const entries = [...await Promise.all(manifest.classes.map(e => readJson('models/skills/'+e.file))),
    ...await Promise.all(extra.classes.map(e => readJson('models/control/'+e.file)))];
  const {descriptors, sources} = await nativeSources();
  const base = Array.from({length: 53}, (_, i) => rotationExp(new Vector3(...resource.codes.slice(i*3, i*3+3))).toArray());
  const results = [], durations = [];
  for (const entry of entries) for (const skill of entry.skills) {
    const test = nativeLayerCase(entry, skill, base, manifest.bones, sources);
    results.push(test.result); durations.push(...test.durations);
    test.layers.reset();
    test.layers.add({...entry, skills: [skill]}, {cycles: 4, hold: 0}, base);
    for (let i = 0; i < test.frames.length; i++) {
      assert.deepEqual(test.layers.sample(base, i ? dt : 0), test.frames[i], 'fixed-source replay must be deterministic');
    }
  }
  const interrupts = entries.map(entry => interruptNative(entry, base, manifest.bones, sources));
  return {classes: entries.length, variants: results.length, sources: descriptors.length,
    frames: durations.length, sampleP95Ms: percentile(durations, .95),
    maxStepRad: Math.max(...results.map(r => r.maxStepRad)),
    cycleBoundaryMaxRad: Math.max(...results.map(r => r.cycleBoundaryMaxRad)),
    interruptionCases: interrupts.length, interruptionMaxStepRad: Math.max(...interrupts),
    rootMotionIncluded: false, repeatedReplayExact: true, unmaskedJointsPreserved: true, results};
}

function percentile(values, p) {
  const sorted = [...values].sort((a, b) => a-b);
  return sorted[Math.floor((sorted.length-1)*p)];
}

function guards(resource, test) {
  const base = quaternions(test.oldLocal.at(-1)).map(q => q.toArray());
  const layers = beatLayers(test.bones, new Map());
  assert.throws(() => layers.sample(base, -dt));
  assert.throws(() => layers.add({skills: [{source: 'missing'}]}, {}, base));
  assert.throws(() => layers.add({skills: []}, {}, base));
  const filter = outputSmoothing(test.bones, resource.settings);
  const point = {q: quaternions(test.local[0]), p: new Vector3(...test.root[0])}, saved = clonePoint(point);
  assert.throws(() => filter.step(point, NaN, null, p => p, finitePoint));
  assert.throws(() => filter.step(point, dt, [NaN], p => p, finitePoint));
  assert.throws(() => filter.step(point, dt, null, p => p, () => false));
  const output = filter.step(point, dt, null, p => p, finitePoint, false);
  assert.deepEqual(output, point);
  assert.deepEqual(saved, point, 'source must remain immutable');
  return {checks: 8, passed: true};
}

function nativeContinuity(resource) {
  const before = beatRuntime(resource.meta, resource.weights, resource.codes);
  const after = beatRuntimeV9(resource.meta, resource.weights, resource.codes);
  let lastRaw = null, lastSmooth = null, switches = 0, rawMax = 0, smoothMax = 0, worstBoundary = null;
  for (let i = 0; i < 2400; i++) {
    const a = before.sample(i ? dt : 0), b = timeCall('nativeResidual', () => after.sample(i ? dt : 0));
    assert.equal(a.token, b.token, 'selection must stay comparable');
    if (lastRaw && a.steps !== lastRaw.steps) {
      switches++;
      for (let j = 0; j < 53; j++) {
        const q = row => rotationExp(new Vector3(...row.values.slice(j*3, j*3+3)));
        rawMax = Math.max(rawMax, q(a).angleTo(q(lastRaw)));
        const gap = q(b).angleTo(q(lastSmooth));
        if (gap > smoothMax) {
          smoothMax = gap; worstBoundary = {bone: resource.meta.bones[j], outgoing: lastSmooth.token,
            incoming: b.token, timeInClip: b.time, stepRad: gap};
        }
      }
    }
    assert.ok(b.values.every(Number.isFinite));
    lastRaw = a; lastSmooth = b;
  }
  return {seconds: 120, switches, baselineBoundaryMaxRad: rawMax, residualBoundaryMaxRad: smoothMax, worstBoundary};
}

async function run() {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) throw new Error('usage: node scripts/hast_sec_runtime.mjs INPUT OUTPUT');
  const fixtures = JSON.parse(await readFile(input, 'utf8')), resource = await resources(), cases = [];
  for (const test of fixtures.cases) {
    cases.push({id: test.id, methods: {hastResidual: residualFrames(test, resource),
      hastLayers: layerFrames(test), hastDamped: dampedFrames(test, resource),
      hastDampedPreserveRoot: dampedFrames(test, resource, true)}});
  }
  const library = await nativeLibrary(resource);
  library.continuity = nativeContinuity(resource);
  const methods = [...new Set(timings.map(t => t.method))];
  const report = {methodology: {modules: ['beat_runtime_v9.mjs', 'beat_layers.mjs', 'v9/output_smoothing.mjs'],
    originalModulesImported: true, solverMap: 'identity except explicit preserve-root variant',
    physicsValidator: 'finite/unit quaternions only; not the HAST full-body feasibility solver',
    skeletonAdapter: '27 independent channels padded to 53 for beatRuntime contract; no skeletal retargeting',
    boundaryClock: 'incoming residual frame zero equals outgoing last frame; subsequent frames advance at 20 Hz',
    dampedHistory: 'continuous full outgoing clip; seam uses actual filtered outputs, not raw history',
    roots: 'XZ translation aligns clip origins; residual/layers preserve incoming root; damped tests both modes'},
    nativeLibrary: library, guards: guards(resource, fixtures.cases[0]),
    timings: Object.fromEntries(methods.map(method => {
      const samples = timings.filter(t => t.method === method).map(t => t.ms);
      return [method, {samples: samples.length, p50Ms: percentile(samples, .5), p95Ms: percentile(samples, .95)}];
    })), cases};
  await writeFile(output, JSON.stringify(report));
  console.log(JSON.stringify({cases: cases.length, libraryClasses: library.classes,
    libraryVariants: library.variants, libraryFrames: library.frames, guards: report.guards}));
}

await run();

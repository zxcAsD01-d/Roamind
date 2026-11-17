import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import { build } from 'esbuild';
import * as THREE from 'three';

await mkdir('output/contact', { recursive: true });
await build({ entryPoints: ['src/contact_tracking.ts'], bundle: true, platform: 'node', format: 'esm',
  outfile: 'output/contact/tracker.mjs', external: ['three', '@pixiv/three-vrm'] });
const { LiveContact, contactWeight } = await import('../output/contact/tracker.mjs');

function rig() {
  const root = new THREE.Group(), bones = new Map();
  const add = (name, parent, xyz) => {
    const node = new THREE.Group(); node.name = name; node.position.fromArray(xyz);
    (bones.get(parent) ?? root).add(node); bones.set(name, node);
  };
  add('hips', null, [0, .92, 0]); add('spine', 'hips', [0, .17, 0]);
  add('chest', 'spine', [0, .20, 0]); add('neck', 'chest', [0, .15, 0]); add('head', 'neck', [0, .12, 0]);
  for (const [side, sign] of [['right', -1], ['left', 1]]) {
    add(side+'UpperArm', 'chest', [sign*.15, .05, 0]);
    add(side+'LowerArm', side+'UpperArm', [sign*.20, -.05, .03]);
    add(side+'Hand', side+'LowerArm', [sign*.05, -.17, .07]);
    add(side+'UpperLeg', 'hips', [sign*.05, -.02, 0]);
    add(side+'LowerLeg', side+'UpperLeg', [0, -.43, 0]);
    add(side+'Foot', side+'LowerLeg', [0, -.43, .01]);
  }
  root.updateWorldMatrix(true, true);
  const humanBones = Object.fromEntries([...bones].map(([name, node]) => [name, { node }]));
  return { root, bones, humanoid: { humanBones, getNormalizedBoneNode: name => bones.get(name) ?? null } };
}

function planFor(vrm, name) {
  const descriptors = {
    right_hand: { bone: 'rightHand', kind: 'arm', radius: .035 },
    left_hand: { bone: 'leftHand', kind: 'arm', radius: .035 },
    head: { bone: 'head', kind: 'axial', radius: .105 },
    right_foot: { bone: 'rightFoot', kind: 'leg', radius: .065 },
    left_foot: { bone: 'leftFoot', kind: 'leg', radius: .065 },
  };
  const descriptor = { name, ...descriptors[name] };
  const center = vrm.bones.get(descriptor.bone).getWorldPosition(new THREE.Vector3());
  const target = { position: center.clone().add(new THREE.Vector3(0, 0, descriptor.radius)).toArray(),
    normal: [0, 0, -1], tangent: [1, 0, 0], personId: 'visual-person', targetId: 'visual-token',
    observedAt: Date.now()/1000, frameId: 1, frame: 'control-world-y-up', sigmaMeters: .01 };
  return { id: 'fixture', frames: 160, effector: descriptor, touch: { target, contactFrames: [30, 130] } };
}

function follow(name, dz) {
  const vrm = rig(), plan = planFor(vrm, name), tracker = new LiveContact(vrm, plan, 0, []);
  const initial = [...vrm.bones.values()].map(bone => bone.quaternion.clone()), poses = [...vrm.bones.values()];
  const observed = { ...plan.touch.target, position: [...plan.touch.target.position], trackingSamples: 2 };
  observed.position[2] += dz; observed.position[1] -= .003;
  tracker.observe(observed);
  const steps = [];
  for (let frame = 60; frame < 110; frame += 1/3) {
    poses.forEach((bone, i) => bone.quaternion.copy(initial[i])); vrm.root.updateWorldMatrix(true, true);
    const report = tracker.update(frame/20, 1/60); steps.push(report);
  }
  const final = steps.at(-1);
  assert.ok(final.displacement >= dz-.004, `${name}: target wasn't followed`);
  assert.ok(final.error < .02, `${name}: large tracking error ${final.error}`);
  assert.deepEqual(vrm.root.position.toArray(), [0, 0, 0]);
  assert.deepEqual(vrm.bones.get('hips').quaternion.toArray(), [0, 0, 0, 1]);
  assert.equal(tracker.update((110-1/3)/20, 1/60), null);
  return { effector: name, displacementMeters: final.displacement,
    maxSolverErrorMeters: Math.max(...steps.map(item => item.error)),
    p95Ms: steps.map(item => item.ms).sort((a, b) => a-b)[Math.floor(steps.length*.95)] };
}

const started = performance.now();
const results = [follow('right_hand', .09), follow('left_hand', .09), follow('head', .025),
  follow('right_foot', .015), follow('left_foot', .015)];
assert.equal(contactWeight(159, 30, 130, 160), 0);
assert.equal(contactWeight(70, 30, 130, 160), 1);
const vrm = rig(), plan = planFor(vrm, 'right_hand');
const live = new LiveContact(vrm, plan, 0, []);
const contact = live.update(3, 1/60);
assert.equal(contact.contactPhase, true);
assert.equal(contact.observedContact, true);
assert.ok(Math.abs(contact.normalGapMeters) < 1e-6);
const missedPlan = structuredClone(plan);
missedPlan.touch.target.position[0] += .10;
const missed = new LiveContact(vrm, missedPlan, 0, []).update(3, 1/60);
assert.ok(missed.error < 1e-6, 'zero solver error must not imply surface contact');
assert.equal(missed.observedContact, false);
assert.ok(missed.tangentMeters > .09);
assert.equal(new LiveContact(vrm, plan, 0, []).update(.5, 1/60).observedContact, false);
assert.throws(() => live.observe({ ...plan.touch.target, personId: 'different' }), /目标/);
assert.throws(() => live.observe({ ...plan.touch.target, targetId: 'different-point' }), /目标/);
const genericPlan = structuredClone(plan);
delete genericPlan.touch.target.personId;
const generic = new LiveContact(vrm, genericPlan, 0, []);
generic.observe(genericPlan.touch.target);
assert.equal(generic.update(3, 1/60).observedContact, true);
assert.throws(() => live.observe({ ...plan.touch.target, observedAt: 0 }), /过期/);
assert.throws(() => live.observe({ ...plan.touch.target, position: [9, 9, 9] }), /范围/);
const point = vrm.bones.get('rightHand').getWorldPosition(new THREE.Vector3());
// A surface exactly one hand radius away is contact, not penetration.
const surfaceBox = { id: 'contact-surface', center: [point.x, point.y, point.z+.085], size: [.1, .1, .1] };
assert.equal(new LiveContact(vrm, plan, 0, [surfaceBox]).update(3, 1/60).observedContact, true);
const penetratingBox = { ...surfaceBox, center: [point.x, point.y, point.z+.075] };
assert.throws(() => new LiveContact(vrm, plan, 0, [penetratingBox]).update(3, 1/60), /骨段碰撞/);
const collision = new LiveContact(vrm, plan, 0, [{ id: 'test-obstacle', center: point.toArray(), size: [.1, .1, .1] }]);
assert.throws(() => collision.update(3, 1/60), /骨段碰撞/);
vrm.bones.get('leftUpperLeg').position.x = 1; vrm.root.updateWorldMatrix(true, true);
assert.throws(() => new LiveContact(vrm, planFor(vrm, 'right_foot'), 0, []).update(3, 1/60), /支撑姿态/);
const report = { fixture: 'synthetic normalized skeleton; not a real VRM precision benchmark',
  results, elapsedMs: performance.now()-started, assertions: 'tracking, non-root motion, bounds, stale/identity/range/collision/support rejection' };
await writeFile('output/contact/tracking-unit.json', JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));

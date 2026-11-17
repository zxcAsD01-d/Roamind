import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, type VRM } from '@pixiv/three-vrm';
import { SmplClip, VrmRetargeter, measureVrmRest, type SmplMotionJson } from './retarget';

type Sample = { local: number[][][]; root: number[][]; oldLocal?: number[][][]; oldRoot?: number[][] };
type Fixture = Sample & { id: string; bones: string[]; parents: number[]; restJoints: number[];
  oldLocal: number[][][]; oldRoot: number[][]; current: Sample };
type Preview = { fixture: Fixture; samples: { methods: Record<string, Sample> } };
type PreviewEntry = { id: string; label: string; file: string };
type Actor = { vrm: VRM; root: THREE.Group; offset: number; driver: VrmRetargeter | null; frames: number };
const element = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`缺少页面元素 ${id}`);
  return node as T;
};
const caseSelect = element<HTMLSelectElement>('case'), methodSelect = element<HTMLSelectElement>('method');
const stage = element<HTMLDivElement>('stage'), slider = element<HTMLInputElement>('time');
const status = element<HTMLDivElement>('status'), errorBox = element<HTMLDivElement>('error');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
stage.append(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#263448');
scene.add(new THREE.HemisphereLight(0xdbeeff, 0x667075, 2.6));
const light = new THREE.DirectionalLight(0xffffff, 2); light.position.set(2, 5, 4); scene.add(light);
scene.add(new THREE.GridHelper(18, 36, 0x6b8d9c, 0x415369));
const camera = new THREE.PerspectiveCamera(40, 1, .05, 80); camera.position.set(0, 2.4, 6.8);
const controls = new OrbitControls(camera, renderer.domElement); controls.target.set(0, .95, 0); controls.update();
let actors: Actor[] = [], entries: PreviewEntry[] = [], preview: Preview | null = null;
let playing = true, elapsed = 0, total = 0, seam = 0, last = performance.now(), serial = 0;

async function readJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`资源读取失败 ${response.status}: ${url}`);
  return await response.json() as T;
}

async function loadActor(offset: number): Promise<Actor> {
  const loader = new GLTFLoader(); loader.register(parser => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync('/models/Wolferia.vrm');
  const vrm = gltf.userData.vrm as VRM | undefined;
  if (!vrm) throw new Error('Wolferia 文件缺少 VRM');
  vrm.scene.rotation.y = Math.PI;
  const bounds = new THREE.Box3().setFromObject(vrm.scene), center = bounds.getCenter(new THREE.Vector3());
  vrm.scene.position.set(-center.x, -bounds.min.y, -center.z);
  const root = new THREE.Group(); root.position.x = offset; root.add(vrm.scene); scene.add(root);
  measureVrmRest(vrm);
  return { vrm, root, offset, driver: null, frames: 0 };
}

function motionFor(sample: Sample, fixture: Fixture): SmplMotionJson {
  const count = fixture.oldLocal.length;
  const prefixLocal = sample.oldLocal ?? fixture.oldLocal, prefixRoot = sample.oldRoot ?? fixture.oldRoot;
  const padding = Math.max(0, count-prefixLocal.length);
  const local = [...Array.from({ length: padding }, () => prefixLocal[0]), ...prefixLocal, ...sample.local];
  const root = [...Array.from({ length: padding }, () => prefixRoot[0]), ...prefixRoot, ...sample.root];
  const origin = root[0];
  return { version: 1, name: fixture.id, source: 'HAST isolated comparison', dataset: 'local-test',
    fps: 20, frames: root.length, duration: root.length/20, up: 'yup', bones: fixture.bones,
    parents: fixture.parents, restJoints: fixture.restJoints, rootOrigin: [0, .92, 0],
    localRot: local.flat(2), rootDelta: root.flatMap(row => row.map((value, i) => value-origin[i])) };
}

function applyPreview(): void {
  if (!preview) return;
  const fixture = preview.fixture, method = methodSelect.value;
  const selected = method === 'raw' ? fixture : preview.samples.methods[method];
  if (!selected) throw new Error('测试结果中没有所选方法');
  for (const [i, sample] of [fixture.current, selected].entries()) {
    const clip = SmplClip.from(motionFor(sample, fixture));
    actors[i].driver = new VrmRetargeter(actors[i].vrm, clip); actors[i].frames = clip.frames;
  }
  seam = fixture.oldLocal.length/20; total = (actors[0].frames-1)/20; elapsed = 0;
  slider.max = String(total);
  element('right-label').textContent = methodSelect.selectedOptions[0].text;
  status.textContent = `${fixture.id} · 接缝 ${seam.toFixed(2)} 秒 · 每帧 0.05 秒`;
  stage.dataset.ready = 'true'; stage.dataset.case = fixture.id;
}

async function selectCase(): Promise<void> {
  const version = ++serial, entry = entries[caseSelect.selectedIndex];
  if (!entry) return;
  const loaded = await readJson<Preview>(`/output/interaction/${entry.file}`);
  if (version !== serial) return;
  preview = loaded; errorBox.textContent = ''; applyPreview();
}

function showError(error: Error): void {
  errorBox.textContent = error.message; stage.dataset.error = error.message;
}

function frame(now: number): void {
  requestAnimationFrame(frame);
  const delta = Math.min(.1, (now-last)/1000); last = now;
  if (playing && total) { elapsed += delta; if (elapsed > total+.8) elapsed = 0; }
  const at = Math.min(elapsed, total); slider.value = String(at);
  element('stamp').textContent = `${at.toFixed(2)}s ${at >= seam ? '· 接缝后' : ''}`;
  for (const actor of actors) if (actor.driver) {
    actor.root.position.copy(actor.driver.rootAt(at*20)); actor.root.position.x += actor.offset;
    actor.driver.apply(at*20); actor.vrm.update(0);
  }
  if (actors.length) {
    const center = actors.reduce((sum, actor) => sum.add(actor.root.position), new THREE.Vector3())
      .divideScalar(actors.length);
    center.y = .95;
    const movement = center.sub(controls.target);
    camera.position.add(movement); controls.target.add(movement); controls.update();
  }
  stage.dataset.time = String(at); renderer.render(scene, camera);
}

new ResizeObserver(() => {
  renderer.setSize(stage.clientWidth, stage.clientHeight);
  camera.aspect = stage.clientWidth/stage.clientHeight; camera.updateProjectionMatrix();
}).observe(stage);
caseSelect.addEventListener('change', () => { selectCase().catch(showError); });
methodSelect.addEventListener('change', () => { try { applyPreview(); } catch (error) {
  showError(error instanceof Error ? error : new Error('切换算法失败'));
} });
element('play').addEventListener('click', () => { playing = !playing; element('play').textContent = playing ? '暂停' : '播放'; });
element('restart').addEventListener('click', () => { elapsed = 0; });
slider.addEventListener('input', () => { playing = false; elapsed = Number(slider.value); element('play').textContent = '播放'; });

async function start(): Promise<void> {
  entries = await readJson<PreviewEntry[]>('/output/interaction/hast-sec-preview-index.json');
  for (const entry of entries) caseSelect.add(new Option(entry.label, entry.id));
  actors = await Promise.all([-1.35, 1.35].map(loadActor));
  await selectCase(); requestAnimationFrame(frame);
}
start().catch(showError);

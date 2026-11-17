import { InteractionBody, type InteractionNode, type BodyOperation } from './interaction_body';
import type { KernelState } from './character';
import { bodyReachProfile, bodyCollisionProfile } from './contact_rig';
import { showMotionTable } from './motion_table_panel';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRM, VRMHumanBoneName, VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { SmplClip, VrmRetargeter, measureVrmRest, type MotionManifestEntry, type SmplMotionJson } from './retarget';
import { capturePlaybackHistory, joinSavedMotion } from './continuity';
import { HouseScene, type HouseWorld } from './house';
import { EgoPerception, MicrophoneCapture, perceptionRequest, type ObservedTarget } from './perception';
import { TargetMarkers, renderTargets } from './targets';
import { TargetFollower } from './follower';
import { AutonomyController } from './behavior';
import { LocomotionLibrary, PRESET_ACTIONS, PresetUnavailableError, PresetCollisionError, presetSeed, type LocomotionStep, type MotionPhase, type MotionSeed, type PresetResult } from './locomotion';
import { ActionLibraryPanel, type LibraryMatch } from './action_library';
import { TouchController } from './touch';
import type { TouchArgs } from './touch';
import { PowerMode } from './power';
import { TextContactController } from './ocr';
import { CharacterRuntime, type CharacterGoal, type BehaviorMemory } from './character';
import { HeadLookController, HEAD_YAW_LIMIT } from './head_look';
import { LumenMotionSystem } from './lumen/runtime';
import { isLumenAction } from './lumen/types';
import { capturePresetBoundary, advancePresetBoundary, type PresetBoundary } from './motion_handoff';
import { ObjectInteractions, type InteractionCatalog } from './object_interactions';
import { MovementProgress } from './movement_progress';
import { workspaceMarkup, bindWorkspaceTabs } from './workspace';
import './style.css';
import './workspace.css';

const movementProgress = new MovementProgress();

type ModelFile = { name: string; file: string; size: string };
type Joint = { key: string; label: string; bone: VRMHumanBoneName };
type ArdyMotion = MotionManifestEntry & { sourceUrl?: string };
type PromptListResponse = { prompts: string[] };
type GeneratedMotionResponse = MotionManifestEntry & { url?: string };
type TrajectoryKeyframe = { time: number; x: number; y: number; z: number; heading: number };
type ArdyKeyframe = { frame: number; root: [number, number, number]; heading: number };
type TimedAction = { action: string; start: number; end: number; destination?: string | number[];
  via?: (string | number[])[]; heading?: number; face?: string | number[] };
type ScenePlan = { id: string; duration: number; frames: number; start: number[]; path: number[][];
  planningMs: number; timeline: LocomotionStep[]; timing?: 'strict' | 'flexible';
  libraryClipId?: string;
  specification?: { steps?: TimedAction[] } };
type ControlCommand = { id: string; kind: 'navigate' | 'execute' | 'interrupt' | 'face' | 'follow' | 'visual' | 'touch' | 'text-contact' | 'gesture' | 'interaction' | 'preset-action'; arguments: {
  prompt?: string;
  node?: InteractionNode; objectId?: string;
  text?: string; direction?: string; occurrence?: number;
  destination?: string | number[]; via?: (string | number[])[]; action?: string; speed?: number;
  duration?: number; planId?: string; clipId?: string; mode?: string;
  heading?: number; face?: string | number[]; targetId?: string; distance?: number; faceTarget?: boolean; behaviorDecisionId?: string; relativeBearing?: number;
  continuationOf?: string;
  continuous?: boolean;
  relativePath?: number[][];
  viewHeading?: number;
  viewPitch?: number;
  goalId?: string;
  selection?: TouchArgs['selection']; hand?: string; operation?: string; amplitude?: number;
} };
type LiveStreamMessage =
  | { type: 'status'; message: string }
  | {
    type: 'chunk'; id: string; name: string; fps: number; frames: number; index: number; startFrame: number;
    totalChunks: number; prompt: string; bones: string[]; parents: number[]; up: 'yup';
    localRot: number[]; rootDelta: number[]; rootOrigin: number[]; generationMs?: number; planId?: string;
    restJoints?: number[]; footContacts?: number[];
    engine?: { mode: string; instances: number; overlapFrames: number; scheduling?: string };
  }
  | { type: 'complete'; entry: MotionManifestEntry & { url: string } }
  | { type: 'error'; error: string };

function isGeneratedMotionResponse(value: GeneratedMotionResponse | { error?: string }): value is GeneratedMotionResponse {
  return 'id' in value && 'file' in value && 'duration' in value;
}

const models: ModelFile[] = await fetch('/api/models').then(async response => {
  if (!response.ok) throw new Error(`模型目录读取失败：HTTP ${response.status}`);
  return await response.json() as ModelFile[];
});
if (!models.length) models.push({ name: '未配置 VRM', file: '', size: '请配置 MOTION_VRM_DIR' });

const joints: Joint[] = [
  { key: 'hips', label: '髋部', bone: VRMHumanBoneName.Hips },
  { key: 'spine', label: '脊柱', bone: VRMHumanBoneName.Spine },
  { key: 'chest', label: '胸部', bone: VRMHumanBoneName.Chest },
  { key: 'head', label: '头部', bone: VRMHumanBoneName.Head },
  { key: 'leftUpperArm', label: '左上臂', bone: VRMHumanBoneName.LeftUpperArm },
  { key: 'leftLowerArm', label: '左前臂', bone: VRMHumanBoneName.LeftLowerArm },
  { key: 'rightUpperArm', label: '右上臂', bone: VRMHumanBoneName.RightUpperArm },
  { key: 'rightLowerArm', label: '右前臂', bone: VRMHumanBoneName.RightLowerArm },
  { key: 'leftUpperLeg', label: '左大腿', bone: VRMHumanBoneName.LeftUpperLeg },
  { key: 'leftLowerLeg', label: '左小腿', bone: VRMHumanBoneName.LeftLowerLeg },
  { key: 'rightUpperLeg', label: '右大腿', bone: VRMHumanBoneName.RightUpperLeg },
  { key: 'rightLowerLeg', label: '右小腿', bone: VRMHumanBoneName.RightLowerLeg },
];

const root = document.querySelector<HTMLDivElement>('#app');
if (!root) throw new Error('App root element was not found');

root.innerHTML = workspaceMarkup;
bindWorkspaceTabs();
const viewport = document.querySelector<HTMLDivElement>('#viewport')!;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#191e20');
const avatarRoot = new THREE.Group();
scene.add(avatarRoot);
const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
camera.position.set(0, 1.35, 4.1);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
viewport.appendChild(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xf0f2ef, 0x45413a, 1.7));
const keyLight = new THREE.DirectionalLight(0xfff4df, 3.4);
keyLight.position.set(-3, 5, 4);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
Object.assign(keyLight.shadow.camera, { left: -15, right: 15, top: 15, bottom: -15, near: .1, far: 60 });
keyLight.shadow.normalBias = .025;
keyLight.position.set(-8, 18, 10);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xd7e6ec, 1.2);
fillLight.position.set(3, 2, -3);
scene.add(fillLight);

const grid = new THREE.GridHelper(12, 24, 0x65726f, 0x343c3a);
grid.position.y = -0.015;
scene.add(grid);
const axes = new THREE.AxesHelper(0.8);
axes.position.y = 0.01;
axes.visible = false;
scene.add(axes);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.9, 0);
controls.enableDamping = true;
controls.minDistance = 1.2;
controls.maxDistance = 9;
controls.maxPolarAngle = Math.PI * 0.91;

const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

let currentVrm: VRM | null = null;
let selectedJoint = joints[0];
let baseRotations = new Map<string, THREE.Euler>();
let pose = { x: 0, y: 0, z: 0 };
let initialPose = new Map<string, THREE.Euler>();
let snapshots: Array<{ name: string; values: Record<string, [number, number, number]> }> = [];
let activeModel = models.find(model => /wolferia/i.test(model.file)) ?? models[0];
let ardyMotions: ArdyMotion[] = [];
let currentMotion: ArdyMotion | null = null;
let currentClip: SmplClip | null = null;
let retargeter: VrmRetargeter | null = null;
let motionTime = 0;
let motionPlaying = false;
const headLook = new HeadLookController();
let activeLook: { command: string; started: number; settledAt: number | null; duration: number } | null = null;
let motionLoopable = false;
let motionLoop = true;
let motionSpeed = 1;
let motionScrubbing = false;
let liveMotionData: SmplMotionJson | null = null;
let liveGenerationActive = false;
let liveUnderflows = 0;
let liveBufferEmpty = false;
let livePriming = false;
const LIVE_START_BUFFER_SECONDS = 2.4;
const liveClientId = crypto.randomUUID();
let generationRevision = 0;
let liveAbort: AbortController | null = null;
let liveStreamId: string | null = null;
let livePrefix: SmplMotionJson | null = null;
let house: HouseScene | null = null;
let houseEnabled = true;
let houseOffset = new THREE.Vector3();
let scenePlanMode = false;
let locomotion: LocomotionLibrary | null = null;
let lumen: LumenMotionSystem | null = null;
let gestureRevision = 0;
const preparingGestures = new Map<string, string>();
let motionMode: 'preset' | 'library' | 'ardy' = 'ardy';
let actionLibrary: ActionLibraryPanel | null = null;
let motionPhases: MotionPhase[] = [];
let sequenceRevision = 0;
let sceneSequence: { step: TimedAction; timing: 'strict' | 'flexible' }[] = [];
let activeCommand: string | null = null;
let activeCommandPending = false;
let activeCommandKind: ControlCommand['kind'] | null = null;
const deferredCompletions: { id: string; atTime: number }[] = [];
const commandReports: { id: string; status: string; detail?: string; retainedCommandId?: string }[] = [];
const seenCommands = new Set<string>();
const seenCommandOrder: string[] = [];
const MAX_SEEN_COMMANDS = 512;
const MAX_CACHED_MOTIONS = 64;
let lastCollision = '';
let sceneReady = false;
let observedTargets: ObservedTarget[] = [];
let follower: TargetFollower | null = null;
let autonomy: AutonomyController | null = null;
let character: CharacterRuntime | null = null;
let objectInteractions: ObjectInteractions | null = null;
let nextAmbientAt = performance.now()+9000;
let ambientFailed = false;
let ego: EgoPerception | null = null;
let touch: TouchController | null = null;
let interactionBody: InteractionBody | null = null;
let textContact: TextContactController | null = null;
let preparingTextCommand: string | null = null;
let touchPartner: THREE.Group | null = null;
let touchPartnerOrigin = new THREE.Vector3();
let touchPartnerTime = 0;
const touchPartnerDirection = new THREE.Vector3(1, 0, 0);
let targetMarkers: TargetMarkers | null = null;
const frameTimes: number[] = [];
let lastPerformanceUpdate = 0;
const followShift = new THREE.Vector3();
let lastAppliedClip: SmplClip | null = null;
let lastAppliedFrame = -1;
const lastAppliedOffset = new THREE.Vector3(Infinity, Infinity, Infinity);
let lastReadoutTime = 0;
let resizeObserver: ResizeObserver;
let trajectoryPoints: TrajectoryKeyframe[] = [
  { time: 0, x: 0, y: 0.92, z: 0, heading: 0 },
  { time: 4, x: 0, y: 0.92, z: 2, heading: 0 },
  { time: 8, x: 2, y: 0.92, z: 4, heading: 45 },
];
const trajectoryVisual = new THREE.Group();
scene.add(trajectoryVisual);

const ardyService = import.meta.env.VITE_MOTION_LAB_URL || 'http://127.0.0.1:8100';

function updateRendererSize(): void {
  const { width, height } = viewport.getBoundingClientRect();
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

resizeObserver = new ResizeObserver(updateRendererSize);
resizeObserver.observe(viewport);
updateRendererSize();

function modelText(value: string): string {
  return value.replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;',
    '"': '&quot;', "'": '&#39;' }[character]!));
}

function createModelCards(): void {
  const list = document.querySelector<HTMLDivElement>('#model-list')!;
  list.innerHTML = models.map((model, index) => `
    <button class="model-card ${model.file === activeModel.file ? 'active' : ''}" data-file="${modelText(model.file)}">
      <span class="model-thumb thumb-${index + 1}"><span class="thumb-orbit"></span><span class="thumb-person">✦</span></span>
      <span class="model-info"><strong>${modelText(model.name)}</strong><small>${modelText(model.file.toUpperCase())}</small></span>
      <span class="model-size">${model.size}</span>
    </button>`).join('');
  list.querySelectorAll<HTMLButtonElement>('.model-card').forEach((card) => {
    card.addEventListener('click', () => {
      const model = models.find((entry) => entry.file === card.dataset.file);
      if (model) void loadModel(model);
    });
  });
}

function createJointControls(): void {
  const select = document.querySelector<HTMLSelectElement>('#joint-select')!;
  select.innerHTML = joints.map((joint) => `<option value="${joint.key}">${joint.label}</option>`).join('');
  select.addEventListener('change', () => {
    selectedJoint = joints.find((joint) => joint.key === select.value) ?? joints[0];
    document.querySelector('#joint-name')!.textContent = selectedJoint.label;
    readPoseFromBone();
    renderAxisControls();
  });
  renderAxisControls();
}

function renderAxisControls(): void {
  const container = document.querySelector<HTMLDivElement>('#axis-controls')!;
  const axesData = [
    { key: 'x' as const, label: 'X', color: 'coral' },
    { key: 'y' as const, label: 'Y', color: 'green' },
    { key: 'z' as const, label: 'Z', color: 'blue' },
  ];
  container.innerHTML = axesData.map((axis) => `
    <label class="axis-row ${axis.color}">
      <span class="axis-name">${axis.label}</span>
      <input type="range" min="-180" max="180" step="1" value="${pose[axis.key]}" data-axis="${axis.key}" aria-label="${selectedJoint.label} ${axis.label} 轴角度">
      <output>${pose[axis.key]}°</output>
    </label>`).join('');
  container.querySelectorAll<HTMLInputElement>('input').forEach((input) => {
    input.addEventListener('input', () => {
      const axis = input.dataset.axis as keyof typeof pose;
      pose[axis] = Number(input.value);
      input.nextElementSibling!.textContent = `${pose[axis]}°`;
      applyJointPose();
      updateReadout();
    });
  });
  updateReadout();
}

function getCurrentBone(): THREE.Object3D | null {
  return currentVrm?.humanoid.getNormalizedBoneNode(selectedJoint.bone) ?? null;
}

function applyJointPose(): void {
  const bone = getCurrentBone();
  const base = baseRotations.get(selectedJoint.key);
  if (!bone || !base) return;
  bone.rotation.set(
    base.x + THREE.MathUtils.degToRad(pose.x),
    base.y + THREE.MathUtils.degToRad(pose.y),
    base.z + THREE.MathUtils.degToRad(pose.z),
    'XYZ',
  );
}

function readPoseFromBone(): void {
  const bone = getCurrentBone();
  const base = baseRotations.get(selectedJoint.key);
  if (!bone || !base) {
    pose = { x: 0, y: 0, z: 0 };
    return;
  }
  pose = {
    x: Math.round(THREE.MathUtils.radToDeg(bone.rotation.x - base.x)),
    y: Math.round(THREE.MathUtils.radToDeg(bone.rotation.y - base.y)),
    z: Math.round(THREE.MathUtils.radToDeg(bone.rotation.z - base.z)),
  };
}

function updateReadout(): void {
  const readout = document.querySelector<HTMLElement>('#angle-readout')!;
  readout.textContent = `X ${pose.x}°　Y ${pose.y}°　Z ${pose.z}°`;
}

function capturePose(): Record<string, [number, number, number]> {
  const values: Record<string, [number, number, number]> = {};
  for (const joint of joints) {
    const bone = currentVrm?.humanoid.getNormalizedBoneNode(joint.bone);
    const base = baseRotations.get(joint.key);
    if (!bone || !base) continue;
    values[joint.key] = [
      THREE.MathUtils.radToDeg(bone.rotation.x - base.x),
      THREE.MathUtils.radToDeg(bone.rotation.y - base.y),
      THREE.MathUtils.radToDeg(bone.rotation.z - base.z),
    ];
  }
  return values;
}

function renderSnapshots(): void {
  const container = document.querySelector<HTMLDivElement>('#snapshots')!;
  document.querySelector('#snapshot-count')!.textContent = String(snapshots.length).padStart(2, '0');
  if (snapshots.length === 0) {
    container.innerHTML = '<p class="empty-state">调整关节后保存快照</p>';
    return;
  }
  container.innerHTML = snapshots.map((snapshot, index) => `
    <div class="snapshot-row"><button class="snapshot-load" data-index="${index}"><span class="snapshot-index">${String(index + 1).padStart(2, '0')}</span><span>${snapshot.name}</span></button><button class="snapshot-delete" data-index="${index}" title="删除快照" aria-label="删除快照">×</button></div>`).join('');
  container.querySelectorAll<HTMLButtonElement>('.snapshot-load').forEach((button) => {
    button.addEventListener('click', () => applySnapshot(snapshots[Number(button.dataset.index)]));
  });
  container.querySelectorAll<HTMLButtonElement>('.snapshot-delete').forEach((button) => {
    button.addEventListener('click', () => {
      snapshots.splice(Number(button.dataset.index), 1);
      renderSnapshots();
    });
  });
}

function applySnapshot(snapshot: (typeof snapshots)[number] | undefined): void {
  if (!snapshot) return;
  for (const joint of joints) {
    const bone = currentVrm?.humanoid.getNormalizedBoneNode(joint.bone);
    const base = baseRotations.get(joint.key);
    const values = snapshot.values[joint.key];
    if (!bone || !base || !values) continue;
    bone.rotation.set(base.x + THREE.MathUtils.degToRad(values[0]), base.y + THREE.MathUtils.degToRad(values[1]), base.z + THREE.MathUtils.degToRad(values[2]), 'XYZ');
  }
  readPoseFromBone();
  renderAxisControls();
}

function resetCurrentJoint(): void {
  pose = { x: 0, y: 0, z: 0 };
  applyJointPose();
  renderAxisControls();
}

function resetAll(): void {
  for (const joint of joints) {
    const bone = currentVrm?.humanoid.getNormalizedBoneNode(joint.bone);
    const initial = initialPose.get(joint.key);
    if (bone && initial) bone.rotation.copy(initial);
  }
  selectedJoint = joints[0];
  document.querySelector<HTMLSelectElement>('#joint-select')!.value = selectedJoint.key;
  document.querySelector('#joint-name')!.textContent = selectedJoint.label;
  pose = { x: 0, y: 0, z: 0 };
  renderAxisControls();
  controls.target.set(0, 0.9, 0);
  camera.position.set(0, 1.35, 4.1);
  controls.update();
}

function fitModel(vrm: VRM): void {
  const bounds = new THREE.Box3().setFromObject(vrm.scene);
  const size = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
  vrm.scene.position.x -= center.x;
  vrm.scene.position.z -= center.z;
  vrm.scene.position.y -= bounds.min.y;
  const height = Math.max(size.y, 1);
  controls.target.set(0, height * 0.48, 0);
  camera.position.set(0, height * 0.72, height * 2.65);
  controls.update();
}

function frameAvatar(vrm: VRM): void {
  const bounds = new THREE.Box3().setFromObject(vrm.scene);
  const height = Math.max(1, bounds.max.y-bounds.min.y);
  const center = bounds.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(0, height*.22, height*2.65));
  controls.update();
}

function updateMotionReadout(): void {
  const duration = currentClip?.motion.duration ?? 0;
  document.querySelector<HTMLInputElement>('#motion-scrub')!.value = duration > 0 ? String(Math.round((motionTime / duration) * 1000)) : '0';
  document.querySelector('#motion-time')!.textContent = `${motionTime.toFixed(1)} s`;
  document.querySelector('#motion-duration-label')!.textContent = `${duration.toFixed(1)} s`;
  document.querySelector<HTMLButtonElement>('#play-motion')!.textContent = motionPlaying || livePriming ? 'Ⅱ' : '▶';
  viewport.dataset.motionTime = motionTime.toFixed(4);
  viewport.dataset.bufferSeconds = Math.max(0, duration - motionTime - 1 / (currentClip?.fps ?? 20)).toFixed(4);
  viewport.dataset.underflows = String(liveUnderflows);
  viewport.dataset.rootX = avatarRoot.position.x.toFixed(4);
  viewport.dataset.rootZ = avatarRoot.position.z.toFixed(4);
  viewport.dataset.motionMode = motionMode;
  viewport.dataset.heading = String(THREE.MathUtils.radToDeg(playbackHeading()));
  viewport.dataset.phase = activeLook ? 'look'
    : motionPhases.find(phase => phase.start <= motionTime && phase.end > motionTime)?.kind ?? 'idle';
  viewport.dataset.clientId = liveClientId;
}

function continuationBrakeIn(): number | null {
  const phase = motionPhases.find(item => item.start <= motionTime && item.end > motionTime);
  if (motionMode !== 'preset' || !motionPlaying || !phase || !['walk', 'start'].includes(phase.kind)) return null;
  const brake = motionPhases.find(item => item.kind === 'stop' && item.start > motionTime);
  return brake ? brake.start-motionTime : null;
}

function visualHandoff(): { headingError: number; tailDistance: number; gazeReady: boolean } {
  const end = currentClip ? currentClip.frames-1 : 0;
  const orientation = currentClip ? new THREE.Quaternion().fromArray(currentClip.local, end*currentClip.joints*4) : null;
  const forward = orientation ? new THREE.Vector3(0, 0, 1).applyQuaternion(orientation) : null;
  const difference = forward ? Math.atan2(forward.x, forward.z)-playbackHeading() : 0;
  const last = retargeter?.rootAt(end).add(houseOffset);
  const settled = activeLook?.settledAt;
  const observation = ego?.visualObservation();
  return { headingError: Math.abs(THREE.MathUtils.radToDeg(Math.atan2(Math.sin(difference), Math.cos(difference)))),
    tailDistance: last ? Math.hypot(last.x-avatarRoot.position.x, last.z-avatarRoot.position.z) : 0,
    gazeReady: settled !== null && settled !== undefined && !!observation?.semanticObservedAt
      && observation.semanticObservedAt >= settled && headLook.settled(playbackHeading()) };
}

function advanceHeadLook(delta: number): void {
  if (!currentVrm) return;
  const phase = motionPhases.find(item => item.start <= motionTime && item.end > motionTime)?.kind;
  headLook.update(currentVrm, delta, playbackHeading(), motionPlaying && (phase === 'walk' || phase === 'start'), motionTime);
  const diagnostic = headLook.diagnostics();
  viewport.dataset.headYaw = diagnostic.yawDegrees.toFixed(3);
  viewport.dataset.headSpeed = diagnostic.speedDegrees.toFixed(3);
  if (!activeLook) return;
  const now = Date.now()/1000;
  if (!headLook.settled(playbackHeading())) activeLook.settledAt = null;
  else activeLook.settledAt ??= now;
  if (activeLook.settledAt !== null && now-activeLook.settledAt >= .35 && now-activeLook.started >= activeLook.duration) {
    activeLook = null; reportCommand('completed');
  } else if (now-activeLook.started > 5) {
    activeLook = null; reportCommand('failed', '头部观察未在安全活动范围内稳定，请转动身体');
  }
}

function renderTrajectoryEditor(): void {
  const container = document.querySelector<HTMLDivElement>('#trajectory-rows')!;
  container.innerHTML = trajectoryPoints.map((point, index) => `
    <div class="trajectory-grid trajectory-row" data-index="${index}">
      <input type="number" required min="0" max="60" step="0.1" value="${point.time}" data-field="time" aria-label="轨迹点 ${index + 1} 时间">
      <input type="number" required step="0.01" value="${point.x}" data-field="x" aria-label="轨迹点 ${index + 1} X 坐标">
      <input type="number" required step="0.01" value="${point.y}" data-field="y" aria-label="轨迹点 ${index + 1} Y 坐标">
      <input type="number" required step="0.01" value="${point.z}" data-field="z" aria-label="轨迹点 ${index + 1} Z 坐标">
      <input type="number" required step="1" value="${point.heading}" data-field="heading" aria-label="轨迹点 ${index + 1} 朝向">
      <button class="trajectory-delete" data-delete="${index}" title="删除轨迹点" aria-label="删除轨迹点 ${index + 1}" ${trajectoryPoints.length <= 2 ? 'disabled' : ''}>×</button>
    </div>`).join('');
  document.querySelector('#trajectory-count')!.textContent = `${String(trajectoryPoints.length).padStart(2, '0')} 点`;
  drawTrajectoryPreview();
}

function drawTrajectoryPreview(): void {
  while (trajectoryVisual.children.length > 0) {
    const child = trajectoryVisual.children[0];
    trajectoryVisual.remove(child);
    if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
      child.geometry.dispose();
      const material = child.material;
      if (Array.isArray(material)) material.forEach((item) => item.dispose());
      else material.dispose();
    }
  }
  const ordered = [...trajectoryPoints].sort((a, b) => a.time - b.time);
  const startX = ordered[0]?.x ?? 0;
  const startZ = ordered[0]?.z ?? 0;
  const positions = ordered.map((point) => new THREE.Vector3(point.x - startX, 0.025, point.z - startZ));
  if (positions.length < 2) return;
  const geometry = new THREE.BufferGeometry().setFromPoints(positions);
  trajectoryVisual.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0xe4ad72 })));
  positions.forEach((position, index) => {
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(index === 0 || index === positions.length - 1 ? 0.075 : 0.055, 12, 8),
      new THREE.MeshBasicMaterial({ color: index === 0 ? 0xb2d7c3 : index === positions.length - 1 ? 0xe4ad72 : 0xe2e8df }),
    );
    marker.position.copy(position);
    trajectoryVisual.add(marker);
  });
}

function addTrajectoryPoint(): void {
  const last = [...trajectoryPoints].sort((a, b) => a.time - b.time).at(-1);
  trajectoryPoints.push({
    time: Math.min(60, (last?.time ?? 0) + 2),
    x: last?.x ?? 0,
    y: last?.y ?? 0.92,
    z: (last?.z ?? 0) + 1,
    heading: last?.heading ?? 0,
  });
  renderTrajectoryEditor();
}

function updateTrajectoryFromInput(target: HTMLInputElement): void {
  const row = target.closest<HTMLElement>('.trajectory-row');
  const point = row ? trajectoryPoints[Number(row.dataset.index)] : undefined;
  const field = target.dataset.field as keyof TrajectoryKeyframe | undefined;
  const value = target.valueAsNumber;
  if (!point || !field || !Number.isFinite(value)) return;
  point[field] = value;
  drawTrajectoryPreview();
}

function getArdyKeyframes(duration: number): ArdyKeyframe[] {
  const ordered = [...trajectoryPoints].sort((a, b) => a.time - b.time);
  return ordered.map((point) => ({
    frame: Math.min(Math.round(point.time * 20), Math.round(duration * 20) - 1),
    root: [point.x, point.y, point.z] as [number, number, number],
    heading: THREE.MathUtils.degToRad(point.heading),
  }));
}

function applyMotionFrame(): void {
  if (interactionBody?.ownsPose) return;
  if (!currentVrm || !currentClip || !retargeter) return;
  const frame = Math.min(currentClip.frames - 1, motionTime * currentClip.fps);
  if (currentClip === lastAppliedClip && frame === lastAppliedFrame && lastAppliedOffset.equals(houseOffset)) return;
  const nextRoot = retargeter.rootAt(frame);
  nextRoot.add(houseOffset);
  if (houseEnabled && house) {
    const collision = house.collisionAt(nextRoot) ?? house.sweep(avatarRoot.position, nextRoot);
    if (collision) {
      motionPlaying = false;
      if (lastCollision !== collision) {
        lastCollision = collision;
        cancelLiveGeneration();
        document.querySelector('#simulation-status')!.textContent = `碰撞已阻止：${collision}；请重新规划或回到出生点`;
        reportCommand('failed', `播放碰撞：${collision}`);
      }
      return;
    }
    lastCollision = '';
  }
  const cameraShift = followShift.copy(nextRoot).sub(avatarRoot.position);
  avatarRoot.position.copy(nextRoot);
  retargeter.apply(frame);
  camera.position.add(cameraShift);
  controls.target.add(cameraShift);
  lastAppliedClip = currentClip;
  lastAppliedFrame = frame;
  lastAppliedOffset.copy(houseOffset);
}

function checkMovementProgress(delta: number): void {
  const phase = viewport.dataset.phase;
  const walking = motionPlaying && activeCommandKind === 'visual' && (phase === 'start' || phase === 'walk');
  if (!movementProgress.update(activeCommand, walking, delta, avatarRoot.position.x, avatarRoot.position.z)) return;
  viewport.dataset.deadlock = JSON.stringify({ command: activeCommand, at: Date.now()/1000,
    x: avatarRoot.position.x, z: avatarRoot.position.z, reason: '持续行走 3 秒位移不足 12cm' });
  reportCommand('failed', '移动死锁：持续行走 3 秒位移不足 12cm，回头换路');
  stopWithAnimation();
}

function advanceMotion(delta: number): void {
  if (!currentClip) return;
  if (motionPlaying) {
    motionTime += delta * motionSpeed;
    const finalTime = Math.max(0, (currentClip.frames - 1) / currentClip.fps);
    if (liveGenerationActive && motionTime >= finalTime) {
      motionTime = finalTime;
      if (!liveBufferEmpty) liveUnderflows++;
      liveBufferEmpty = true;
    } else if (motionTime >= currentClip.motion.duration) {
      if (liveGenerationActive) {
        motionTime = Math.max(0, currentClip.motion.duration - 1 / currentClip.fps);
      } else if (motionLoop && motionLoopable) {
        motionTime %= currentClip.motion.duration;
      } else {
        motionTime = Math.max(0, currentClip.motion.duration - 1 / currentClip.fps);
        motionPlaying = false;
        if (sceneSequence.length) void runSceneTask(continueSceneSequence, activeCommand);
        else if (!activeLook && !follower?.active && !activeCommandPending && preparingTextCommand !== activeCommand) reportCommand('completed');
      }
      updateMotionReadout();
    }
  }
  while (deferredCompletions.length && motionTime >= deferredCompletions[0].atTime) {
    const completed = deferredCompletions.shift()!;
    commandReports.push({ id: completed.id, status: 'completed' });
  }
  applyMotionFrame();
  if (!motionScrubbing && performance.now()-lastReadoutTime > 100) {
    updateMotionReadout(); lastReadoutTime = performance.now();
  }
}

function setMotionPlaying(playing: boolean): void {
  if (livePriming && !playing) {
    livePriming = false;
    motionPlaying = false;
    updateMotionReadout();
    return;
  }
  if (playing && houseEnabled && sceneReady && !scenePlanMode) {
    document.querySelector('#ardy-status')!.textContent = '房屋内请通过移动或时间规划生成带碰撞约束的动作';
    return;
  }
  if (!currentClip) {
    document.querySelector('#ardy-status')!.textContent = '先选择一条 ARDY 动作';
    return;
  }
  motionPlaying = playing;
  livePriming = false;
  updateMotionReadout();
}

function renderMotionOptions(selectedId = currentMotion?.id ?? ''): void {
  const select = document.querySelector<HTMLSelectElement>('#motion-select')!;
  select.innerHTML = ardyMotions.map((motion) => `<option value="${motion.id}">${motion.name} · ${motion.duration.toFixed(1)}s</option>`).join('');
  if (selectedId) select.value = selectedId;
  document.querySelector('#motion-count')!.textContent = String(ardyMotions.length).padStart(2, '0');
}

function cacheMotion(entry: ArdyMotion): void {
  ardyMotions = [entry, ...ardyMotions.filter((item) => item.id !== entry.id)].slice(0, MAX_CACHED_MOTIONS);
}

async function loadMotion(entry: ArdyMotion, continuous = false): Promise<void> {
  if (houseEnabled && sceneReady) {
    document.querySelector('#ardy-status')!.textContent = '已有动作未经过当前房屋的碰撞校验；关闭房屋仿真后可播放';
    renderMotionOptions();
    return;
  }
  reportCommand('interrupted', '切换已有动作');
  cancelLiveGeneration();
  scenePlanMode = false;
  const revision = generationRevision;
  const status = document.querySelector<HTMLElement>('#ardy-status')!;
  status.textContent = `加载动作 ${entry.name}…`;
  try {
    const url = entry.sourceUrl ?? `/ardy/${encodeURIComponent(entry.file)}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`动作读取失败：HTTP ${response.status}`);
    const json = await response.json() as SmplMotionJson;
    if (revision !== generationRevision) return;
    let clip = SmplClip.from(json);
    let nextTime = 0;
    const resume = continuous && motionPlaying;
    if (continuous && currentClip) {
      const history = capturePlaybackHistory(currentClip, motionTime, false, motionSpeed);
      clip = SmplClip.from(joinSavedMotion(history.motion, clip));
      nextTime = history.time;
    }
    currentMotion = entry;
    currentClip = clip;
    retargeter = currentVrm ? new VrmRetargeter(currentVrm, clip) : null;
    const last = clip.rootAt(clip.frames - 1, new THREE.Vector3());
    motionLoopable = !continuous && last.length() < 1.5;
    motionLoop = motionLoopable;
    document.querySelector<HTMLInputElement>('#motion-loop')!.checked = motionLoop;
    document.querySelector<HTMLInputElement>('#motion-loop')!.disabled = !motionLoopable;
    motionTime = nextTime;
    motionPlaying = resume;
    applyMotionFrame();
    updateMotionReadout();
    renderMotionOptions(entry.id);
    status.textContent = `${clip.frames} 帧 · ${clip.fps} fps · ARDY Y-up`;
  } catch (error) {
    if (revision !== generationRevision) return;
    status.textContent = error instanceof Error ? error.message : '动作加载失败';
  }
}

async function initArdyLibrary(): Promise<void> {
  const status = document.querySelector<HTMLElement>('#ardy-status')!;
  const promptSelect = document.querySelector<HTMLSelectElement>('#prompt-select')!;
  try {
    const manifestResponse = await fetch('/ardy/manifest.json');
    if (!manifestResponse.ok) throw new Error(`动作列表读取失败：HTTP ${manifestResponse.status}`);
    const manifest = await manifestResponse.json() as ArdyMotion[];
    const initial = manifest.find((entry) => entry.id === 'ardy_walk_long');
    ardyMotions = (initial ? [initial, ...manifest.filter((entry) => entry.id !== initial.id)] : manifest)
      .slice(0, MAX_CACHED_MOTIONS);
    renderMotionOptions();
    if (ardyMotions.length > 0) {
      await loadMotion(ardyMotions[0]);
    }
  } catch (error) {
    const detail = error instanceof Error ? error.message : 'ARDY 动作读取失败';
    status.textContent = detail;
  }

  try {
    const promptResponse = await fetch(`${ardyService}/api/prompts`);
    if (!promptResponse.ok) throw new Error(`ARDY 服务 HTTP ${promptResponse.status}`);
    const promptList = await promptResponse.json() as PromptListResponse;
    promptSelect.innerHTML = promptList.prompts.map((prompt) => `<option value="${prompt}">${prompt}</option>`).join('');
    status.textContent = `${ardyMotions.length} 条本地动作 · ${promptList.prompts.length} 个缓存提示词`;
  } catch (error) {
    const detail = error instanceof Error ? error.message : 'ARDY 服务连接失败';
    status.textContent = ardyMotions.length > 0 ? `${ardyMotions.length} 条本地动作可播放 · ${detail}` : detail;
    promptSelect.innerHTML = '<option value="">ARDY 服务未连接</option>';
  }
}

function setGenerationBusy(busy: boolean): void {
  document.querySelectorAll<HTMLButtonElement>('#generate-motion, #generate-trajectory')
    .forEach((button) => { button.disabled = busy; });
}

function cancelLiveGeneration(): void {
  generationRevision++;
  const controller = liveAbort;
  liveAbort = null;
  controller?.abort();
  liveGenerationActive = false;
  liveStreamId = null;
  livePriming = false;
  if (!controller) return;
  document.querySelector('#ardy-status')!.textContent = '生成已中断，已接收动作保留';
  void fetch(`${ardyService}/api/generate/cancel`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ clientId: liveClientId, revision: generationRevision }),
  }).then((response) => {
    if (!response.ok) throw new Error(`取消生成失败：HTTP ${response.status}`);
  }).catch((error: Error) => { console.warn('ARDY 取消请求失败', error); });
}

function prepareLiveContinuation(): void {
  motionMode = 'ardy'; motionPhases = [];
  livePrefix = null;
  liveMotionData = null;
  if (currentClip) {
    const sceneHistory = scenePlanMode && locomotion;
    const history = sceneHistory ? currentPresetSeed()
      : capturePlaybackHistory(currentClip, motionTime, motionPlaying, motionSpeed);
    const croppedSeconds = motionTime - history.time;
    deferredCompletions.forEach(report => { report.atTime -= croppedSeconds; });
    livePrefix = history.motion;
    if (scenePlanMode && !sceneHistory) {
      const source = history.motion.rootDelta;
      for (let i = 0; i < source.length; i += 3) {
        source[i] += houseOffset.x; source[i+1] += houseOffset.y; source[i+2] += houseOffset.z;
      }
      houseOffset.set(0, 0, 0);
    }
    if (sceneHistory) houseOffset.set(0, 0, 0);
    currentClip = SmplClip.from(history.motion);
    motionTime = history.time;
    retargeter = currentVrm ? new VrmRetargeter(currentVrm, currentClip) : null;
  }
  motionLoop = false;
  motionLoopable = false;
  document.querySelector<HTMLInputElement>('#motion-loop')!.checked = false;
  document.querySelector<HTMLInputElement>('#motion-loop')!.disabled = true;
  liveGenerationActive = true;
  liveBufferEmpty = false;
  liveUnderflows = 0;
  livePriming = !currentClip || !motionPlaying;
  motionPlaying = !livePriming;
}

async function generateArdyMotion(): Promise<void> {
  if (houseEnabled && house) {
    document.querySelector('#ardy-status')!.textContent = '房屋模式请使用移动或时间规划，让碰撞参与生成；自由生成可关闭房屋仿真';
    return;
  }
  const status = document.querySelector<HTMLElement>('#ardy-status')!;
  const prompt = document.querySelector<HTMLSelectElement>('#prompt-select')!.value;
  const duration = Number(document.querySelector<HTMLInputElement>('#generation-duration')!.value);
  if (!prompt || !Number.isFinite(duration) || duration < 1 || duration > 60) {
    status.textContent = '请选择缓存提示词，并将时长设为 1–60 秒';
    return;
  }

  cancelLiveGeneration();
  const revision = ++generationRevision;
  const controller = new AbortController();
  liveAbort = controller;
  status.textContent = '正在从当前动作续生成，可更换提示词再次点击切换…';
  try {
    prepareLiveContinuation();
    const response = await fetch(`${ardyService}/api/generate/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, duration, history: livePrefix, clientId: liveClientId, revision }),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) throw new Error(`实时生成请求失败：HTTP ${response.status}`);
    await readLiveStream(response.body, revision, status);
  } catch (error) {
    if (revision !== generationRevision || controller.signal.aborted) return;
    cancelLiveGeneration();
    status.textContent = error instanceof Error ? error.message : 'ARDY 生成失败';
  } finally {
    if (revision !== generationRevision && !controller.signal.aborted) controller.abort();
    if (revision === generationRevision) liveAbort = null;
  }
}

async function readLiveStream(body: ReadableStream<Uint8Array>, revision: number, status: HTMLElement): Promise<void> {
  const reader = body.getReader();
  try {
    const decoder = new TextDecoder();
    let buffer = '';
    let streamCompleted = false;
    while (true) {
      const { value, done } = await reader.read();
      if (revision !== generationRevision) { await reader.cancel(); return; }
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        if (!line.trim()) continue;
        const message = JSON.parse(line) as LiveStreamMessage;
        handleLiveStreamMessage(message, status);
        streamCompleted ||= message.type === 'complete';
      }
      if (done) break;
    }
    if (buffer.trim()) {
      const message = JSON.parse(buffer) as LiveStreamMessage;
      handleLiveStreamMessage(message, status);
      streamCompleted ||= message.type === 'complete';
    }
    if (!streamCompleted) throw new Error('实时生成连接提前结束');
  } finally {
    reader.releaseLock();
  }
}

function handleLiveStreamMessage(message: LiveStreamMessage, status: HTMLElement): void {
  if (message.type === 'status') {
    status.textContent = message.message;
    return;
  }
  if (message.type === 'error') throw new Error(message.error);
  if (message.type === 'chunk') {
    appendLiveMotionChunk(message);
    const engine = message.engine;
    const detail = engine?.mode === 'dual-overlap'
      ? `双模型${engine.scheduling === 'alternating' ? '交替接力' : ''} · ${engine.overlapFrames / message.fps}s 重叠融合`
      : '连贯生成';
    status.textContent = `连贯生成中：${(liveMotionData?.duration ?? 0).toFixed(1)} 秒 · ${detail}`;
    return;
  }

  liveGenerationActive = false;
  void actionLibrary?.refresh().catch((error: Error) => { console.warn('动作已生成，但动作库列表刷新失败', error); });
  const generated: ArdyMotion = {
    ...message.entry,
    sourceUrl: `${ardyService}/files/${encodeURIComponent(message.entry.file)}`,
  };
  cacheMotion(generated);
  currentMotion = generated;
  renderMotionOptions(generated.id);
  status.textContent = `生成完成并已保存：${generated.duration.toFixed(1)} 秒`;
  if (livePriming && currentClip) {
    livePriming = false;
    motionPlaying = true;
    updateMotionReadout();
    if (scenePlanMode) reportCommand('playing');
  }
}

function appendLiveMotionChunk(chunk: Extract<LiveStreamMessage, { type: 'chunk' }>): void {
  if (liveStreamId !== chunk.id) {
    if (liveStreamId) throw new Error('实时动作流标识发生变化');
    beginLiveMotion(chunk);
  }

  const motion = liveMotionData;
  if (!motion || !currentMotion) throw new Error('实时动作缓冲尚未初始化');
  if (chunk.startFrame !== motion.frames) throw new Error('实时动作分段顺序错误');
  motion.localRot.push(...chunk.localRot);
  motion.rootDelta.push(...chunk.rootDelta);
  if (chunk.footContacts) {
    motion.footContacts ??= new Array<number>(motion.frames*4).fill(0);
    motion.footContacts.push(...chunk.footContacts);
  }
  motion.frames += chunk.frames;
  liveBufferEmpty = false;
  motion.duration = motion.frames / motion.fps;
  currentMotion = {
    ...currentMotion,
    frames: motion.frames,
    duration: motion.duration,
  };
  if (currentClip?.motion === motion) {
    currentClip.appendFrames();
  } else {
    currentClip = SmplClip.from(motion);
    retargeter = currentVrm ? new VrmRetargeter(currentVrm, currentClip) : null;
  }
  applyMotionFrame();
  updateMotionReadout();
  const bufferedSeconds = Math.max(0, (motion.frames-1)/motion.fps-motionTime);
  if (livePriming && bufferedSeconds >= LIVE_START_BUFFER_SECONDS) {
    livePriming = false;
    motionPlaying = true;
    document.querySelector('#ardy-status')!.textContent = `已预缓冲 ${bufferedSeconds.toFixed(1)} 秒 · 开始平滑播放`;
  }
  if (scenePlanMode && motionPlaying) reportCommand('playing');
}

function beginLiveMotion(chunk: Extract<LiveStreamMessage, { type: 'chunk' }>): void {
  liveStreamId = chunk.id;
  const prefix = livePrefix;
  liveMotionData = {
    version: 1,
    name: chunk.name,
    source: 'ardy-service-live',
    dataset: `service · ${chunk.prompt}`,
    up: 'yup',
    fps: chunk.fps,
    frames: prefix?.frames ?? 0,
    duration: prefix?.duration ?? 0,
    bones: chunk.bones,
    parents: chunk.parents,
    localRot: prefix ? [...prefix.localRot] : [],
    rootDelta: prefix ? [...prefix.rootDelta] : [],
    rootOrigin: chunk.rootOrigin,
    restJoints: chunk.restJoints,
    footContacts: prefix?.footContacts ? [...prefix.footContacts] : undefined,
    prompt: chunk.prompt,
  };
  currentMotion = {
    id: chunk.id,
    file: '',
    name: chunk.name,
    dataset: liveMotionData.dataset,
    fps: chunk.fps,
    frames: 0,
    duration: 0,
    bytes: 0,
    prompt: chunk.prompt,
  };
  if (!prefix) motionTime = 0;
  motionLoop = false;
  motionLoopable = false;
  document.querySelector<HTMLInputElement>('#motion-loop')!.checked = false;
  document.querySelector<HTMLInputElement>('#motion-loop')!.disabled = true;
}

async function generateTrajectoryMotion(): Promise<void> {
  if (houseEnabled) {
    document.querySelector('#ardy-status')!.textContent = '房屋模式请使用途经房间或时间规划；旧轨迹编辑器用于关闭房屋后的自由研究';
    return;
  }
  const prompt = document.querySelector<HTMLSelectElement>('#prompt-select')!.value;
  const duration = Number(document.querySelector<HTMLInputElement>('#generation-duration')!.value);
  const status = document.querySelector<HTMLElement>('#ardy-status')!;
  if (!prompt || !Number.isFinite(duration) || duration < 1 || duration > 60) {
    status.textContent = '请选择缓存提示词，并将时长设为 1–60 秒';
    return;
  }
  const inputs = document.querySelectorAll<HTMLInputElement>('#trajectory-rows input');
  if ([...inputs].some((input) => !input.reportValidity())) {
    status.textContent = '请填写有效的轨迹时间、坐标与朝向';
    return;
  }
  if (trajectoryPoints.some((point) => point.time < 0 || point.time > duration)) {
    status.textContent = '轨迹点时间不能超过动作时长';
    return;
  }
  const keyframes = getArdyKeyframes(duration);
  if (keyframes.length < 2) {
    status.textContent = '至少需要两个位于生成时长内的轨迹点';
    return;
  }
  if (new Set(keyframes.map((keyframe) => keyframe.frame)).size !== keyframes.length) {
    status.textContent = '轨迹点时间不能重复';
    return;
  }
  setGenerationBusy(true);
  cancelLiveGeneration();
  const revision = generationRevision;
  status.textContent = `ARDY 按 ${keyframes.length} 个根轨迹点自回归生成中…`;
  try {
    const response = await fetch(`${ardyService}/api/keyframes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, duration, keyframes }),
    });
    const result = await response.json() as GeneratedMotionResponse | { error?: string };
    if (revision !== generationRevision) return;
    if (!response.ok || !isGeneratedMotionResponse(result)) {
      throw new Error('error' in result ? result.error ?? `HTTP ${response.status}` : `HTTP ${response.status}`);
    }
    const generated: ArdyMotion = { ...result, sourceUrl: `${ardyService}/files/${encodeURIComponent(result.file)}` };
    cacheMotion(generated);
    renderMotionOptions(generated.id);
    await loadMotion(generated, true);
    if (currentMotion?.id !== generated.id) throw new Error('轨迹已提交，但生成动作未能加载');
    setMotionPlaying(true);
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : '轨迹生成失败';
  } finally {
    setGenerationBusy(false);
  }
}

async function loadModel(model: ModelFile): Promise<void> {
  if (!model.file) {
    document.querySelector('#load-status')!.textContent = '请将有授权的 VRM 放入 models/avatars，或设置 MOTION_VRM_DIR 后重启网页服务';
    return;
  }
  interactionBody?.reset();
  gestureRevision++; preparingGestures.clear(); lumen?.cancel('切换角色');
  activeLook = null; headLook.release();
  touch?.cancel();
  const overlay = document.querySelector<HTMLDivElement>('#load-overlay')!;
  const status = document.querySelector<HTMLElement>('#load-status')!;
  const toast = document.querySelector<HTMLDivElement>('#error-toast')!;
  overlay.classList.add('visible');
  overlay.querySelector('span')!.textContent = `正在加载 ${model.name}`;
  toast.classList.remove('visible');
  status.textContent = `加载 ${model.name}…`;
  document.querySelectorAll('.model-card').forEach((card) => card.classList.toggle('active', (card as HTMLButtonElement).dataset.file === model.file));
  try {
    if (character?.active) {
      await sceneRequest('/api/goals/cancel', { clientId: liveClientId });
      autonomy?.suspend();
    }
    const gltf = await loader.loadAsync(`/models/${encodeURIComponent(model.file)}`);
    const vrm = gltf.userData.vrm as VRM | undefined;
    if (!vrm) throw new Error('文件中没有可识别的 VRM 数据');
    if (currentVrm) {
      VRMUtils.deepDispose(currentVrm.scene);
      avatarRoot.remove(currentVrm.scene);
    }
    currentVrm = vrm;
    vrm.scene.traverse(node => { if (node instanceof THREE.Mesh) { node.castShadow = true; node.receiveShadow = true; } });
    ego?.resetOrigin();
    currentVrm.scene.rotation.y = Math.PI;
    if (!houseEnabled || !sceneReady) avatarRoot.position.set(0, 0, 0);
    fitModel(currentVrm);
    avatarRoot.add(currentVrm.scene);
    measureVrmRest(currentVrm);
    house?.setBodyProfile(bodyCollisionProfile(currentVrm));
    [locomotion, lumen] = await Promise.all([LocomotionLibrary.load(currentVrm), LumenMotionSystem.load(currentVrm)]);
    showMotionTable(locomotion.table);
    baseRotations = new Map();
    initialPose = new Map();
    for (const joint of joints) {
      const bone = currentVrm.humanoid.getNormalizedBoneNode(joint.bone);
      if (!bone) continue;
      baseRotations.set(joint.key, bone.rotation.clone());
      initialPose.set(joint.key, bone.rotation.clone());
    }
    activeModel = model;
    ambientFailed = false;
    document.querySelector('#stage-model')!.textContent = model.name.toUpperCase();
    snapshots = [];
    renderSnapshots();
    resetAll();
    retargeter = currentClip ? new VrmRetargeter(currentVrm, currentClip) : null;
    applyMotionFrame();
    status.textContent = `${model.name} 已就绪 · 步幅比例 ${locomotion.strideScale.toFixed(3)}`;
  } catch (error) {
    const message = error instanceof Error ? error.message : '模型加载失败';
    status.textContent = '模型加载失败';
    toast.textContent = `${message}。检查模型文件是否仍在桌面/文件/VRM 文件夹中。`;
    toast.classList.add('visible');
  } finally {
    overlay.classList.remove('visible');
  }
}

let simulationAccumulator = 0;
const simulationStep = 1 / 60;
let nextRenderAt = 0;
let renderedFrames = 0;
let simulationSteps = 0;

function animate(now = performance.now()): void {
  requestAnimationFrame(animate);
  const interval = power.fps ? 1000/power.fps : 0;
  if (interval && now+.5 < nextRenderAt) return;
  nextRenderAt = interval ? now+interval-Math.max(0, now-nextRenderAt)%interval : now;
  timer.update();
  const delta = timer.getDelta();
  advanceTouchPartner(delta);
  simulationAccumulator += Math.min(delta, .1);
  while (simulationAccumulator >= simulationStep) {
    headLook.restore();
    lumen?.restore();
    if (!interactionBody?.ownsPose) advanceMotion(simulationStep);
    interactionBody?.update(simulationStep);
    checkMovementProgress(simulationStep);
    lumen?.update(simulationStep);
    touch?.update(motionTime, simulationStep, motionPlaying);
    advanceHeadLook(simulationStep);
    currentVrm?.update(simulationStep);
    simulationSteps++;
    simulationAccumulator -= simulationStep;
  }
  controls.update();
  renderer.render(scene, camera);
  renderedFrames++;
  measurePerformance(delta);
  targetMarkers?.update();
}

async function sceneRequest<T>(path: string, body?: object, timeoutMs = 15000): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${ardyService}${path}`, body ? {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: controller.signal,
    } : { signal: controller.signal });
    const result = await response.json() as T | { error: string };
    if (!response.ok || (typeof result === 'object' && result !== null && 'error' in result)) {
      throw new Error(typeof result === 'object' && result !== null && 'error' in result ? result.error : `HTTP ${response.status}`);
    }
    return result as T;
  } catch (error) {
    if (controller.signal.aborted) throw new Error(`仿真服务请求超时：${path}`);
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function rememberCommand(id: string): boolean {
  if (seenCommands.has(id)) return false;
  seenCommands.add(id); seenCommandOrder.push(id);
  while (seenCommandOrder.length > MAX_SEEN_COMMANDS) {
    const expired = seenCommandOrder.shift();
    if (expired) seenCommands.delete(expired);
  }
  return true;
}

function reportCommand(status: string, detail = ''): void {
  if (status === 'completed' && (activeCommandKind === 'touch' || activeCommandKind === 'text-contact')) {
    const failure = touch?.completionError();
    if (failure) { status = 'failed'; detail = failure; }
  }
  if ((activeCommandKind === 'touch' || activeCommandKind === 'text-contact') && ['completed', 'failed', 'interrupted'].includes(status)) {
    touch?.finish(status, detail);
    if (activeCommandKind === 'text-contact') textContact?.finish(status, detail);
    touch?.cancel();
  }
  if (['playing', 'generating', 'completed', 'failed', 'interrupted'].includes(status)) activeCommandPending = false;
  if (['failed', 'interrupted'].includes(status)) { sceneSequence = []; sequenceRevision++; }
  if (['completed', 'failed', 'interrupted'].includes(status)) follower?.stop();
  if (status === 'failed' || status === 'interrupted') {
    for (const report of deferredCompletions.splice(0)) commandReports.push({ id: report.id, status, detail });
  }
  if (!activeCommand) return;
  if (status === 'playing' && commandReports.at(-1)?.status === 'playing'
    && commandReports.at(-1)?.id === activeCommand) return;
  commandReports.push({ id: activeCommand, status, detail });
  if (['completed', 'failed', 'interrupted'].includes(status)) { activeCommand = null; activeCommandKind = null; }
}

function resetHouse(preserveCommand = false): void {
  interactionBody?.reset();
  gestureRevision++; preparingGestures.clear(); lumen?.cancel('重置位置');
  activeLook = null; headLook.release();
  touch?.cancel();
  cancelLiveGeneration();
  ego?.resetOrigin();
  if (!preserveCommand) reportCommand('interrupted');
  scenePlanMode = false;
  motionPlaying = false;
  motionTime = 0;
  motionPhases = [];
  if (house && locomotion) {
    const seed = locomotion.idle(new THREE.Vector3(house.world.spawn[0], 0, house.world.spawn[2]), 0);
    currentClip = SmplClip.from(seed.motion);
    retargeter = currentVrm ? new VrmRetargeter(currentVrm, currentClip) : null;
    motionMode = 'preset'; houseOffset.set(0, 0, 0);
  }
  if (house && currentClip) {
    const first = currentClip.rootAt(0, new THREE.Vector3());
    houseOffset.set(house.world.spawn[0]-first.x, -first.y, house.world.spawn[2]-first.z);
    lastCollision = '';
    avatarRoot.position.set(house.world.spawn[0], 0, house.world.spawn[2]);
    applyMotionFrame();
  }
  updateMotionReadout();
  document.querySelector('#simulation-status')!.textContent = '出生点已重置 · MCP 在线';
}

function planningStart(retainTail = false): number[] {
  if (!currentClip) return house?.world.spawn ?? [0, .92, 0];
  if (locomotion) {
    const seed = presetSeed(currentClip, motionTime, motionPlaying, houseOffset, retainTail);
    const last = seed.motion.rootDelta.slice(-3);
    return [last[0], last[1]+.92, last[2]];
  }
  const history = capturePlaybackHistory(currentClip, motionTime, motionPlaying, motionSpeed);
  const index = (history.motion.frames-1)*3;
  return [history.motion.rootDelta[index]+houseOffset.x,
    history.motion.rootDelta[index+1]+houseOffset.y+.92, history.motion.rootDelta[index+2]+houseOffset.z];
}

function showPlan(plan: ScenePlan): void {
  house?.showPath(plan.path);
  const list = document.querySelector<HTMLOListElement>('#plan-timeline')!;
  list.replaceChildren(...plan.timeline.map(step => {
    const item = document.createElement('li');
    item.textContent = `${step.start.toFixed(2)}–${step.end.toFixed(2)} s · ${step.action} · ${step.distance.toFixed(1)} m`;
    return item;
  }));
  document.querySelector('#simulation-status')!.textContent = `路线校验通过 · ${plan.duration.toFixed(2)} 秒 · 规划 ${plan.planningMs} ms`;
}

async function planRoute(execute: boolean): Promise<void> {
  if (execute) autonomy?.suspend();
  const destination = document.querySelector<HTMLSelectElement>('#destination-select')!.value;
  const via = document.querySelector<HTMLInputElement>('#route-via')!.value.split(/[,，]/).map(s => s.trim()).filter(Boolean);
  await runSceneTask(async () => {
    const fixedHeading = document.querySelector<HTMLInputElement>('#move-facing')!.checked;
    const plan = await sceneRequest<ScenePlan>('/api/plans/navigation', { start: planningStart(), destination, via,
      startHeading: THREE.MathUtils.radToDeg(playbackHeading()),
      ...(fixedHeading ? { heading:Number(document.querySelector<HTMLInputElement>('#facing-angle')!.value) } : {}) });
    showPlan(plan);
    if (execute) await sceneRequest('/api/sim/execute', { planId: plan.id, clientId: liveClientId });
  });
}

async function planTimeline(execute: boolean): Promise<void> {
  if (execute) autonomy?.suspend();
  await runSceneTask(async () => {
    const steps = JSON.parse(document.querySelector<HTMLTextAreaElement>('#plan-editor')!.value) as object[];
    const plan = await sceneRequest<ScenePlan>('/api/plans/timeline', { start: planningStart(), steps,
      startHeading:THREE.MathUtils.radToDeg(playbackHeading()) });
    showPlan(plan);
    if (execute) await sceneRequest('/api/sim/execute', { planId: plan.id, clientId: liveClientId });
  });
}

async function runSceneTask(task: () => Promise<void>, commandId: string | null = null,
  recoverPreparation?: (message: string) => boolean): Promise<void> {
  try { await task(); }
  catch (error) {
    if (commandId && commandId !== activeCommand) return;
    const message = error instanceof Error ? error.message : '场景操作失败';
    document.querySelector('#simulation-status')!.textContent = message;
    document.querySelector('#character-state')!.textContent = message;
    if (recoverPreparation?.(message)) return;
    if (commandId) {
      cancelLiveGeneration(); motionPlaying = false; updateMotionReadout(); reportCommand('failed', message);
    }
  }
}

async function executeScenePlan(plan: ScenePlan, previousCommand?: string, skipCache = false, boundary?: PresetBoundary): Promise<void> {
  touch?.cancel();
  if (!houseEnabled) throw new Error('请先开启房屋仿真');
  const steps = plan.specification?.steps;
  if (steps && steps.length > 1 && !canUsePreset(plan)) {
    sceneSequence = []; sequenceRevision++;
    let previous = 0;
    for (const step of steps) {
      if (step.start > previous) sceneSequence.push({ step: { action: 'observe', start: 0, end: step.start-previous }, timing: 'strict' });
      sceneSequence.push({ step: { ...step, start: 0, end: step.end-step.start }, timing: plan.timing ?? 'strict' });
      previous = step.end;
    }
    await continueSceneSequence(); return;
  }
  if (canUsePreset(plan)) {
    try { playPresetPlan(plan, previousCommand, boundary); return; }
    catch (error) {
      if (!(error instanceof PresetUnavailableError)) throw error;
      if (activeCommandKind === 'visual') throw new Error('基础步态质量未通过：当前预设组合不安全，改换路线；不激活 ARDY 代替普通移动');
    }
  }
  const command = activeCommand, generation = generationRevision;
  const seed = currentPresetSeed();
  const capturedTime = motionTime;
  const stored = skipCache ? null : await sceneRequest<LibraryMatch>('/api/library/resolve', {
    planId: plan.id, history: seed.motion, ...(plan.libraryClipId ? { clipId: plan.libraryClipId } : {}),
  });
  if (command !== activeCommand || generation !== generationRevision) return;
  if (stored?.hit && stored.motion) {
    seed.time += Math.max(0, motionTime-capturedTime);
    if (seed.time > (seed.motion.frames-1)/20+.001) {
      const rebased = await sceneRequest<ScenePlan>('/api/plans/rebase', { planId: plan.id,
        start: planningStart(), startHeading: THREE.MathUtils.radToDeg(playbackHeading()) });
      if (command !== activeCommand || generation !== generationRevision) return;
      await executeScenePlan(rebased, previousCommand, true); return;
    }
    installPreset({ motion: stored.motion, seconds: stored.motion.duration-seed.time,
      phases: [{ kind: 'action', start: seed.motion.duration, end: stored.motion.duration,
        source: 'validated-action-library', variant: stored.clipId }] }, seed, 'library');
    document.querySelector('#simulation-status')!.textContent = `播放库 ${stored.clipId} · ${stored.variants} 套变体 · 未激活 ARDY`;
    void actionLibrary?.refresh().catch((error: Error) => { console.warn('动作库播放记录刷新失败', error); });
    if (previousCommand) deferredCompletions.push({ id: previousCommand, atTime: (seed.motion.frames-1)/20 });
    return;
  }
  motionMode = 'ardy'; motionPhases = [];
  cancelLiveGeneration();
  const revision = ++generationRevision;
  const controller = new AbortController();
  liveAbort = controller;
  scenePlanMode = true;
  reportCommand('generating');
  try {
    prepareLiveContinuation();
    if (previousCommand && livePrefix) {
      deferredCompletions.push({ id: previousCommand, atTime: (livePrefix.frames - 1) / livePrefix.fps });
    }
    const response = await fetch(`${ardyService}/api/generate/stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
      body: JSON.stringify({ prompt: 'A person walks forward.', duration: 1, planId: plan.id,
        history: livePrefix, clientId: liveClientId, revision }),
    });
    if (!response.ok || !response.body) throw new Error(`规划生成失败 HTTP ${response.status}`);
    await readLiveStream(response.body, revision, document.querySelector<HTMLElement>('#simulation-status')!);
    if (revision === generationRevision) document.querySelector('#simulation-status')!.textContent = '规划生成完成 · 正在按时间表播放';
  } catch (error) {
    if (revision !== generationRevision || controller.signal.aborted) return;
    cancelLiveGeneration();
    throw error;
  } finally {
    if (revision === generationRevision) liveAbort = null;
  }
}

function canUsePreset(plan: ScenePlan): boolean {
  return plan.timeline.every(step => {
    if (!PRESET_ACTIONS.has(step.action)) return false;
    if (!step.fixedHeading || step.heading === undefined || step.distance < .01) return true;
    const path = step.path ?? plan.path;
    return path.slice(1).every((point, index) => {
      const direction = Math.atan2(point[0]-path[index][0], point[2]-path[index][2]);
      const difference = Math.atan2(Math.sin(direction-step.heading!), Math.cos(direction-step.heading!));
      return Math.abs(difference) < .3;
    });
  });
}

async function continueSceneSequence(): Promise<void> {
  const next = sceneSequence.shift();
  if (!next) return;
  const revision = sequenceRevision;
  if (next.step.action === 'observe') {
    playPresetPlan({ id: 'timeline_wait', duration: next.step.end, frames: Math.round(next.step.end*20),
      start: planningStart(), path: [], planningMs: 0, timing: next.timing,
      timeline: [{ ...next.step, distance: 0 }] });
    return;
  }
  const plan = await sceneRequest<ScenePlan>('/api/plans/timeline', { start: planningStart(),
    startHeading: THREE.MathUtils.radToDeg(playbackHeading()), steps: [next.step], timing: next.timing });
  if (revision !== sequenceRevision) return;
  showPlan(plan); await executeScenePlan(plan);
}

function currentPresetSeed(retainTail = false): MotionSeed {
  if (!locomotion) throw new Error('基础动画尚未就绪');
  if (!currentClip) return locomotion.idle(avatarRoot.position, playbackHeading());
  return capturePresetBoundary(currentClip, motionTime, motionPlaying, houseOffset, motionPhases, retainTail).seed;
}

function safePresetRoot(previous: THREE.Vector3, next: THREE.Vector3): string | null {
  return houseEnabled && house ? house.collisionAt(next) ?? house.sweep(previous, next) : null;
}

function installPreset(result: PresetResult, seed: MotionSeed, mode: 'preset' | 'library' | 'ardy' = 'preset'): void {
  const cropped = motionTime-seed.time;
  const prefixEnd = seed.motion.frames/20;
  const prefixPhases = motionPhases.map(phase => ({ ...phase, start: Math.max(0, phase.start-cropped),
    end: Math.min(prefixEnd, phase.end-cropped) })).filter(phase => phase.end > phase.start);
  deferredCompletions.forEach(report => { report.atTime -= cropped; });
  cancelLiveGeneration();
  currentClip = SmplClip.from(result.motion);
  retargeter = currentVrm ? new VrmRetargeter(currentVrm, currentClip) : null;
  houseOffset.set(0, 0, 0); motionTime = seed.time;
  motionMode = mode; motionPhases = [...prefixPhases, ...result.phases]; scenePlanMode = true;
  motionPlaying = true; motionSpeed = 1; motionLoopable = false; motionLoop = false;
  document.querySelector<HTMLSelectElement>('#motion-speed')!.value = '1';
  document.querySelector<HTMLInputElement>('#motion-loop')!.checked = false;
  document.querySelector<HTMLInputElement>('#motion-loop')!.disabled = true;
  lastAppliedClip = null;
  applyMotionFrame(); updateMotionReadout(); reportCommand('playing');
}

function compileExplorationPlan(plan: ScenePlan, seed: MotionSeed): PresetResult {
  if (!locomotion) throw new Error('基础动画尚未就绪');
  try { return locomotion.compile(plan, seed, safePresetRoot); }
  catch (error) {
    const step = plan.timeline[0];
    if (!(error instanceof PresetCollisionError) || activeCommandKind !== 'visual'
      || plan.timeline.length !== 1 || !step || step.action !== 'forward' || step.path?.length) throw error;
    // Probe complete natural steps against physical collision, never centimetre nudges.
    for (const distance of [2, 1.2, .7]) {
      if (distance >= step.distance-.1) continue;
      const shorter = { ...plan, duration: distance/.5,
        timeline: [{ ...step, distance, end: distance/.5 }] };
      try { return locomotion.compile(shorter, seed, safePresetRoot); }
      catch (retryError) {
        if (!(retryError instanceof PresetCollisionError)) throw retryError;
      }
    }
    throw error;
  }
}

function playPresetPlan(plan: ScenePlan, previousCommand?: string, boundary?: PresetBoundary): void {
  if (!locomotion) throw new Error('基础动画尚未就绪');
  const turning = motionPhases.some(phase => phase.kind === 'turn' && phase.start <= motionTime && phase.end > motionTime);
  const seed = boundary ? advancePresetBoundary(boundary, currentClip, motionTime)
    : currentPresetSeed(!!previousCommand && turning);
  const result = compileExplorationPlan(plan, seed);
  installPreset(result, seed);
  if (previousCommand) deferredCompletions.push({ id: previousCommand, atTime: (seed.motion.frames-1)/20 });
  document.querySelector('#simulation-status')!.textContent = result.table
    ? `动作表连续执行 · ${result.table.entries.length} 段 · 路线偏差 ≤ ${(result.table.pathMaxError*100).toFixed(1)}cm · ${result.seconds.toFixed(2)} 秒`
    : `预设低速步态 / 迈步转身 · 实际 ${result.seconds.toFixed(2)} 秒 · ARDY 未激活`;
  document.querySelector('#motion-table-result')!.textContent = result.table
    ? `全身接缝 RMS ${(result.table.seamPoseRms*180/Math.PI).toFixed(1)}° · 终点误差 ${(result.table.goalError*100).toFixed(1)}cm`
    : locomotion.tableStatus ? `动作表覆盖不足，已采用基础步态：${locomotion.tableStatus}` : '';
}

function stopWithAnimation(): void {
  if (interactionBody?.ownsPose) { motionPlaying = false; return; }
  cancelLiveGeneration();
  if (!locomotion || !currentClip || !motionPlaying) { motionPlaying = false; updateMotionReadout(); return; }
  const seed = currentPresetSeed();
  try { installPreset(locomotion.stop(seed, safePresetRoot), seed); }
  catch (error) {
    motionPlaying = false; updateMotionReadout();
    document.querySelector('#simulation-status')!.textContent = error instanceof Error ? error.message : '停步动画失败';
  }
}

function playAmbientIdle(): void {
  if (!document.querySelector<HTMLInputElement>('#ambient-idle')?.checked || ambientFailed || !locomotion) return;
  if (interactionBody?.ownsPose || motionPlaying || lumen?.active || preparingGestures.size > 0 || activeCommand || liveGenerationActive || autonomy?.active || character?.active || touch?.active || follower?.active) {
    nextAmbientAt = performance.now()+9000; return;
  }
  if (performance.now() < nextAmbientAt || !sceneReady || !houseEnabled) return;
  nextAmbientAt = performance.now()+12000;
  try {
    const seed = currentPresetSeed();
    const result = locomotion.compile({ id: 'ambient', duration: 2, path: [], timing: 'flexible',
      timeline: [{ action: 'observe', start: 0, end: 2, distance: 0 }] }, seed, safePresetRoot);
    installPreset(result, seed);
  } catch (error) {
    ambientFailed = true;
    document.querySelector('#simulation-status')!.textContent = `自然待机已停用：${error instanceof Error ? error.message : String(error)}`;
    console.error('自然待机未通过校验', error);
  }
}

async function executeLumenGesture(command: ControlCommand): Promise<void> {
  const revision = gestureRevision;
  const side = command.arguments.action?.startsWith('lumen:left-') ? 'left' : 'right';
  preparingGestures.set(side, command.id);
  nextAmbientAt = performance.now()+9000;
  if (!command.arguments.behaviorDecisionId || command.arguments.goalId) autonomy?.suspend();
  try {
    if (!lumen) throw new Error('流明动作系统尚未就绪');
    const action = command.arguments.action;
    if (!isLumenAction(action)) throw new Error('流明动作不受支持');
    commandReports.push({ id: command.id, status: 'received' });
    const prepared = await lumen.prepare(action, command.arguments.duration ?? 3);
    if (revision !== gestureRevision || preparingGestures.get(side) !== command.id) {
      commandReports.push({ id: command.id, status: 'interrupted', detail: '准备期间动作被替换' }); return;
    }
    lumen.start(command.id, prepared);
  } catch (error) {
    commandReports.push({ id: command.id, status: 'failed', detail: error instanceof Error ? error.message : '流明动作准备失败' });
  } finally { if (preparingGestures.get(side) === command.id) preparingGestures.delete(side); }
}

/** Turn with the native locomotion system before lowering onto an observed support. */
async function alignRestNode(command: string, node: InteractionNode): Promise<void> {
  if (Date.now()/1000-node.observedAt > 2) throw new Error('坐卧节点过期，请重新观察');
  const target = new THREE.Vector3().fromArray(node.controlPoint);
  const heading = Math.atan2(avatarRoot.position.x-target.x, avatarRoot.position.z-target.z);
  const delta = Math.atan2(Math.sin(heading-playbackHeading()), Math.cos(heading-playbackHeading()));
  if (Math.abs(delta) < THREE.MathUtils.degToRad(12)) return;
  const plan: ScenePlan = { id: `node-align-${command}`, start: planningStart(), duration: 3, frames: 60,
    planningMs: 0, path: [], timing: 'flexible', timeline: [{ action: delta > 0 ? 'turn_left' : 'turn_right',
      start: 0, end: 3, distance: 0, heading, fixedHeading: true }] };
  playPresetPlan(plan);
  activeCommandPending = true;
  const deadline = performance.now()+8000;
  while (motionPlaying && activeCommand === command) {
    if (performance.now() > deadline) throw new Error('坐下前转身未完成');
    await new Promise<void>(resolve => setTimeout(resolve, 30));
  }
}

async function processCommand(command: ControlCommand): Promise<void> {
  if (command.kind === 'gesture') { await executeLumenGesture(command); return; }
  if (['touch', 'text-contact', 'interrupt', 'interaction', 'preset-action'].includes(command.kind)) {
    gestureRevision++; preparingGestures.clear(); lumen?.cancel('执行接触或停止命令');
  }
  if (!command.arguments.behaviorDecisionId || command.arguments.goalId) autonomy?.suspend();
  nextAmbientAt = performance.now()+9000;
  ambientFailed = false;
  touch?.cancel();
  await interactionBody?.cancel();
  follower?.stop();
  const continuation = command.kind === 'visual' && command.arguments.continuationOf === activeCommand
    && activeCommandKind === 'visual' ? activeCommand : null;
  const rollback = continuation ? { id: continuation, kind: activeCommandKind,
    pending: activeCommandPending, clip: currentClip, look: activeLook } : null;
  const previousLook = continuation && activeLook?.command === continuation;
  activeLook = null;
  if (!continuation) reportCommand('interrupted', '被新命令替换');
  if (command.kind !== 'visual') headLook.release();
  cancelLiveGeneration();
  activeCommand = command.id;
  activeCommandPending = true;
  activeCommandKind = command.kind;
  reportCommand('received');
  await runSceneTask(async () => {
    const args = command.arguments;
    if (command.kind === 'preset-action') {
      if (!locomotion || !args.prompt) throw new Error('完整动作表或动作语义尚未就绪');
      if (interactionBody?.snapshot().supported) await interactionBody.execute(command.id, 'stand');
      if (activeCommand !== command.id) return;
      const seed = currentPresetSeed();
      installPreset(locomotion.tableAction(args.prompt, seed, safePresetRoot), seed, 'preset');
      document.querySelector('#simulation-status')!.textContent = '原生 AMASS 动作 · 已通过当前骨架与接缝质量检查';
      return;
    }
    if (command.kind === 'interaction') {
      if (!interactionBody || !['sit', 'lie', 'pickup', 'stand', 'release', 'play_computer', 'eat', 'drink'].includes(args.operation ?? '')) throw new Error('未知节点交互操作');
      if (!interactionBody.ownsPose) await stabilizeTouch(() => activeCommand === command.id);
      if (activeCommand !== command.id) return;
      if (args.operation === 'play_computer' && interactionBody.snapshot().supported) await interactionBody.execute(command.id, 'stand');
      if (activeCommand !== command.id) return;
      if ((args.operation === 'sit' || args.operation === 'lie') && args.node) await alignRestNode(command.id, args.node);
      if (args.operation === 'play_computer' && args.node?.seat) await alignRestNode(command.id, args.node.seat);
      if (activeCommand !== command.id) return;
      if (['play_computer', 'eat', 'drink'].includes(args.operation ?? '')) headLook.release();
      motionPlaying = false; activeCommandPending = true;
      await interactionBody.execute(command.id, args.operation as BodyOperation, args.objectId, args.node, args.duration);
      if (activeCommand === command.id) reportCommand('completed', '支撑/持有执行状态已确认');
      return;
    }
    if (command.kind !== 'interrupt' && interactionBody?.snapshot().supported) {
      motionPlaying = false; await interactionBody.execute(command.id, 'stand');
      if (activeCommand !== command.id) return;
      lastAppliedClip = null;
    }
    if (command.kind === 'text-contact') {
      if (!textContact || !args.text) throw new Error('文字触碰控制器或目标文字尚未就绪');
      preparingTextCommand = command.id;
      try { await textContact.execute({ ...args, text: args.text }, () => activeCommand === command.id); }
      finally { if (preparingTextCommand === command.id) preparingTextCommand = null; }
      return;
    }
    if (command.kind === 'touch') {
      if (!touch || !args.selection) throw new Error('触摸控制器或视觉目标尚未就绪');
      await touch.execute({ ...args, selection: args.selection }, () => activeCommand === command.id);
      return;
    }
    if (command.kind === 'follow') {
      if (!args.targetId) throw new Error('跟随缺少目标编号');
      follower?.start({ ...args, targetId:args.targetId });
      return;
    }
    if (command.kind === 'visual') {
      if (args.action === 'look') {
        const target = THREE.MathUtils.degToRad(args.viewHeading ?? THREE.MathUtils.radToDeg(ego?.navigationHeading() ?? playbackHeading()))
          -THREE.MathUtils.degToRad(args.relativeBearing ?? 0);
        const angle = Math.atan2(Math.sin(target-playbackHeading()), Math.cos(target-playbackHeading()));
        if (Math.abs(angle) > THREE.MathUtils.degToRad(HEAD_YAW_LIMIT)) throw new Error('观察方向超出头颈范围，需要先转身');
        headLook.aim(target, playbackHeading(), args.viewPitch ?? 0);
        activeLook = { command: command.id, started: Date.now()/1000, settledAt: null, duration: args.duration ?? 1.2 };
        if (continuation && !previousLook) deferredCompletions.push({ id: continuation, atTime: (currentClip!.frames-1)/20 });
        if (previousLook) commandReports.push({ id: continuation!, status: 'completed' });
        reportCommand('playing'); return;
      }
      const boundary = locomotion && currentClip ? capturePresetBoundary(currentClip, motionTime, motionPlaying,
        houseOffset, motionPhases, !!continuation && viewport.dataset.phase === 'turn', !!continuation) : undefined;
      const plan = await sceneRequest<ScenePlan>('/api/plans/visual-step', {
        start: boundary?.start ?? planningStart(), startHeading: THREE.MathUtils.radToDeg(boundary?.heading ?? playbackHeading()),
        viewHeading: args.viewHeading ?? THREE.MathUtils.radToDeg(ego?.navigationHeading() ?? playbackHeading()),
        action: args.action, duration: args.duration, relativeBearing: args.relativeBearing, distance: args.distance,
        continuous: args.continuous, relativePath: args.relativePath,
      }, continuation ? 1200 : 15000);
      if (activeCommand !== command.id) return;
      headLook.explore();
      showPlan(plan);
      await executeScenePlan(plan, previousLook ? undefined : continuation ?? undefined, false, boundary);
      if (previousLook) commandReports.push({ id: continuation!, status: 'completed' });
      return;
    }
    if (command.kind === 'interrupt') {
      if (args.mode === 'reset') resetHouse(true);
      else if (args.mode === 'resume') { motionPlaying = true; updateMotionReadout(); }
      else { stopWithAnimation(); if (!motionPlaying) reportCommand('completed'); return; }
      reportCommand('completed');
      return;
    }
    const path = command.kind === 'face' ? '/api/plans/timeline' : command.kind === 'navigate' ? '/api/plans/navigation' : '/api/plans/rebase';
    const plan = await sceneRequest<ScenePlan>(path,
      { ...args, start: planningStart(), startHeading:THREE.MathUtils.radToDeg(playbackHeading()),
        ...(command.kind === 'face' ? { steps:[{action:'walk',start:0,end:args.duration ?? 2,
          ...(args.face !== undefined ? {face:args.face} : {heading:args.heading})}], timing: 'flexible' } : {}) });
    if (activeCommand !== command.id) return;
    if (args.clipId) plan.libraryClipId = args.clipId;
    showPlan(plan);
    await executeScenePlan(plan);
  }, command.id, rollback ? message => {
    if (currentClip !== rollback.clip) return false;
    commandReports.push({ id: command.id, status: 'failed', detail: message, retainedCommandId: rollback.id });
    activeCommand = rollback.id; activeCommandKind = rollback.kind;
    activeCommandPending = rollback.pending;
    activeLook = rollback.look;
    if (!motionPlaying && !activeLook) reportCommand('completed');
    else commandReports.push({ id: rollback.id, status: 'playing', detail: '续接准备失败，继续播放当前动作' });
    return true;
  } : undefined);
}

async function syncSimulation(): Promise<void> {
  commandReports.push(...(lumen?.drainReports() ?? []));
  const reports = commandReports.splice(0);
  try {
    const observation = ego?.visualObservation() ?? null;
    const result = await sceneRequest<{ commands: ControlCommand[]; targets:ObservedTarget[];
      goal: CharacterGoal | null; behavior: BehaviorMemory; interactions: InteractionCatalog; kernel: KernelState }>('/api/sim/sync', {
      clientId: liveClientId, reports, vision: observation, state: { position: [avatarRoot.position.x, avatarRoot.position.y+.92, avatarRoot.position.z],
        interaction: interactionBody?.snapshot(),
        body: currentVrm ? { ...bodyReachProfile(currentVrm, avatarRoot),
          cameraPitch: Number(document.querySelector<HTMLInputElement>('#ego-pitch')?.value ?? 0) } : null,
        heading: THREE.MathUtils.radToDeg(playbackHeading()), following:follower?.active ?? false,
        time: motionTime, playing: !!interactionBody?.busy || motionPlaying || !!activeLook || !!lumen?.active, generating: liveGenerationActive || preparingGestures.size > 0, houseEnabled, sceneReady,
        bufferedSeconds: activeLook ? .4 : Math.max(0, (currentClip?.motion.duration ?? 0) - motionTime - 1 / (currentClip?.fps ?? 20)),
        locomotion: { mode: motionMode, strideScale: locomotion?.strideScale, phase: activeLook ? 'look' : viewport.dataset.phase,
          actionSystem: lumen?.state(), handoff: visualHandoff(), headLook: headLook.diagnostics(),
          quality: locomotion?.lastQuality,
          brakeIn: continuationBrakeIn(),
          calibration: locomotion?.calibration,
          phases: motionPhases }, perceptionClientId: ego?.sessionId, perception: ego?.diagnostics(),
        fps: frameTimes.length ? 1000/(frameTimes.reduce((a,b)=>a+b,0)/frameTimes.length) : 0 },
    });
    observedTargets = result.targets; renderTargets(observedTargets);
    objectInteractions?.sync(result.interactions);
    if (result.kernel) character?.syncKernel(result.kernel);
    character?.syncComputer(interactionBody?.snapshot().computer ?? null);
    character?.sync(result.goal, result.behavior, observation, { playing: motionPlaying || !!activeLook || !!lumen?.active, generating: liveGenerationActive || preparingGestures.size > 0,
      activeCommand, phase: viewport.dataset.phase ?? 'idle' });
    autonomy?.update({ playing: motionPlaying || !!activeLook || !!lumen?.active, generating: liveGenerationActive || preparingGestures.size > 0, following: follower?.active ?? false,
      buffered: activeLook ? .4 : (currentClip?.motion.duration ?? 0) - motionTime - 1 / (currentClip?.fps ?? 20),
      handoffReady: visualHandoff().gazeReady,
      brakeIn: continuationBrakeIn(),
      visualCommandId: activeCommandKind === 'visual' ? activeCommand : null });
    for (const command of result.commands) {
      if (!rememberCommand(command.id)) continue;
      void processCommand(command);
    }
    playAmbientIdle();
  } catch (error) {
    commandReports.unshift(...reports);
    const message = error instanceof Error ? error.message : '仿真连接失败';
    document.querySelector('#simulation-status')!.textContent = message;
    document.querySelector('#character-state')!.textContent = `仿真连接中断 · ${message}`;
    character?.disconnected(message);
  } finally {
    const delay = character?.active || activeCommand || commandReports.length ? 150 : 500;
    window.setTimeout(() => { void syncSimulation(); }, delay);
  }
}

async function initHouse(): Promise<void> {
  await runSceneTask(async () => {
    const world = await sceneRequest<HouseWorld>('/api/world');
    house?.dispose();
    house = new HouseScene(world, scene);
    if (currentVrm) house.setBodyProfile(bodyCollisionProfile(currentVrm));
    document.querySelector<HTMLSelectElement>('#destination-select')!.replaceChildren(...world.rooms.map(room => {
      const option = document.createElement('option'); option.value = room.id; option.textContent = room.name; return option;
    }));
    resetHouse(); house.frame(camera, controls); grid.visible = false;
    sceneReady = true;
    initAutonomyControls();
    character = new CharacterRuntime({ request: sceneRequest, clientId: liveClientId,
      enableVision: () => ego?.setEnabled(true), suspendAutonomy: () => autonomy?.suspend(),
      ready: () => sceneReady && !!locomotion && !!currentVrm });
    objectInteractions = new ObjectInteractions({ request: sceneRequest, clientId: liveClientId,
      submitted: () => { autonomy?.suspend(); ego?.setEnabled(true); } });
    ego?.setEnabled(true);
    void syncSimulation();
  });
}

function measurePerformance(delta: number): void {
  if (delta > 0 && delta < .5) { frameTimes.push(delta*1000); if (frameTimes.length > 240) frameTimes.shift(); }
  if (performance.now()-lastPerformanceUpdate < 1000 || !frameTimes.length) return;
  lastPerformanceUpdate = performance.now();
  const sorted = [...frameTimes].sort((a,b)=>a-b);
  const p95 = sorted[Math.floor(sorted.length*.95)];
  const fps = 1000/(frameTimes.reduce((a,b)=>a+b,0)/frameTimes.length);
  document.querySelector('#performance-status')!.textContent = `${fps.toFixed(1)} fps · p95 ${p95.toFixed(1)} ms · ${renderer.info.render.calls} draw · ${(renderer.info.render.triangles/1000).toFixed(0)}k 三角形`;
  document.querySelector<HTMLElement>('#performance-status')!.dataset.metrics = JSON.stringify({sampledAt:performance.now(),fps,p95,calls:renderer.info.render.calls,triangles:renderer.info.render.triangles,
    powerSaving:power.enabled, active:power.active, targetFps:power.fps, renderedFrames, simulationSteps, perception:ego?.diagnostics()});
}

function playbackHeading(): number {
  if (!currentClip || currentClip.frames < 1) return 0;
  const frame = Math.min(currentClip.frames-1,Math.floor(motionTime*currentClip.fps));
  const rotation = new THREE.Quaternion().fromArray(currentClip.world,frame*currentClip.joints*4);
  const forward = new THREE.Vector3(0,0,1).applyQuaternion(rotation);
  return Math.atan2(forward.x,forward.z);
}

async function stabilizeTouch(current: () => boolean): Promise<void> {
  stopWithAnimation();
  reportCommand('generating'); activeCommandPending = true;
  const deadline = performance.now()+6000;
  while (motionPlaying && current()) {
    if (performance.now() > deadline) throw new Error('停步未完成，无法稳定触摸');
    await new Promise<void>(resolve => window.setTimeout(resolve, 50));
  }
}

async function toggleTouchPartner(): Promise<void> {
  touch?.cancel();
  if (touchPartner) {
    scene.remove(touchPartner); VRMUtils.deepDispose(touchPartner); touchPartner = null; return;
  }
  if (!locomotion || !currentVrm) throw new Error('请先加载模型和站立动作');
  const gltf = await loader.loadAsync(`/models/${encodeURIComponent(activeModel.file)}`);
  const vrm = gltf.userData.vrm as VRM | undefined;
  if (!vrm) { VRMUtils.deepDispose(gltf.scene); throw new Error('测试角色缺少 VRM 数据'); }
  const group = new THREE.Group(); group.name = 'visual-touch-partner';
  const savedCamera = camera.position.clone(), savedTarget = controls.target.clone();
  vrm.scene.rotation.y = Math.PI; fitModel(vrm);
  camera.position.copy(savedCamera); controls.target.copy(savedTarget); controls.update();
  group.add(vrm.scene); scene.add(group);
  measureVrmRest(vrm);
  const yaw = playbackHeading(), position = avatarRoot.position.clone().add(new THREE.Vector3(Math.sin(yaw)*.75, 0, Math.cos(yaw)*.75));
  const seed = locomotion.idle(position, yaw+Math.PI), clip = SmplClip.from(seed.motion);
  const driver = new VrmRetargeter(vrm, clip);
  group.position.copy(driver.rootAt(0)); driver.apply(0); vrm.update(0);
  group.updateWorldMatrix(true, true);
  touchPartner = group;
  touchPartnerOrigin.copy(group.position); touchPartnerTime = 0;
  touchPartnerDirection.set(1, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
  document.querySelector<HTMLElement>('#touch-partner')!.dataset.position = JSON.stringify(group.position.toArray());
  document.querySelector('#touch-status')!.textContent = '测试角色已添加 · 触摸位置仍只从 RGB 双目图像估计';
}

function advanceTouchPartner(delta: number): void {
  if (!touchPartner || !document.querySelector<HTMLInputElement>('#touch-partner-moving')?.checked) return;
  touchPartnerTime += Math.min(delta, .1);
  const value = Number(document.querySelector<HTMLInputElement>('#touch-partner-range')?.value ?? .08);
  const range = Number.isFinite(value) ? THREE.MathUtils.clamp(value, .05, 1.5) : .08;
  const offset = touchPartnerDirection.clone().multiplyScalar(range*Math.sin(touchPartnerTime*.9));
  touchPartner.position.copy(touchPartnerOrigin).add(offset); touchPartner.updateWorldMatrix(true, true);
  document.querySelector<HTMLElement>('#touch-partner')!.dataset.position = JSON.stringify(touchPartner.position.toArray());
}

async function prepareTextView(heading: number | undefined, current: () => boolean): Promise<void> {
  if (heading !== undefined) {
    const plan = await sceneRequest<ScenePlan>('/api/plans/timeline', { start: planningStart(),
      startHeading: THREE.MathUtils.radToDeg(playbackHeading()), timing: 'flexible',
      steps: [{ action: 'walk', start: 0, end: 2, heading }] });
    if (!current()) return;
    await executeScenePlan(plan);
  } else stopWithAnimation();
  const deadline = performance.now()+30000;
  while (current() && motionPlaying) {
    if (performance.now() > deadline) throw new Error('文字识别前转向停步超时');
    await new Promise<void>(resolve => setTimeout(resolve, 50));
  }
  if (current()) { reportCommand('generating'); activeCommandPending = true; }
}

function initPerceptionControls(): void {
  interactionBody = new InteractionBody({ vrm: () => currentVrm, root: avatarRoot,
    obstacles: () => houseEnabled && house ? house.world.objects : [], props: () => house?.props ?? null,
    computers: () => houseEnabled && house ? house.computers : [] });
  ego = new EgoPerception({scene,renderer,avatar:avatarRoot,vrm:()=>currentVrm,house:()=>house,
    heading:playbackHeading,observe:body=>sceneRequest('/api/targets/observe',body),
    cadence: () => power.cadence,
    navigating: () => !!(character?.navigating || autonomy?.active),
  });
  touch = new TouchController({ ego: () => ego, request: sceneRequest, root: avatarRoot, vrm: () => currentVrm,
    seed: currentPresetSeed, stabilize: stabilizeTouch,
    play: (motion, seed) => installPreset({ motion, seconds: motion.duration-seed.time,
      phases: [{ kind: 'action', start: seed.motion.duration, end: motion.duration, source: 'visual-touch' }] }, seed, 'ardy'),
    stop: reason => { reportCommand('failed', reason); stopWithAnimation(); }, addPartner: toggleTouchPartner,
    obstacles: () => houseEnabled && house ? house.world.objects : [] });
  textContact = new TextContactController({ ego: () => ego, touch: () => touch, prepare: prepareTextView,
    request: sceneRequest, scene, avatar: avatarRoot, heading: playbackHeading, vrm: () => currentVrm });
  targetMarkers = new TargetMarkers(scene,renderer.domElement,camera,
    body=>sceneRequest('/api/targets/observe',body),position=>!!house && house.collisionAt(position)===null);
  follower = new TargetFollower({targets:()=>observedTargets,
    playback:()=>({buffered:(currentClip?.motion.duration ?? 0)-motionTime,playing:motionPlaying,generating:liveGenerationActive}),
    hold:()=>{stopWithAnimation();},
    generate:async (options, stillCurrent)=>{
      const plan=await sceneRequest<ScenePlan>('/api/plans/tracking',{...options,start:planningStart(),startHeading:THREE.MathUtils.radToDeg(playbackHeading())});
      if(!stillCurrent()) return;
      showPlan(plan);await executeScenePlan(plan);
    },
    status:message=>{document.querySelector('#tracking-status')!.textContent=message;},
    failed:message=>{document.querySelector('#tracking-status')!.textContent=message;reportCommand('failed',message);},
  });
  bindTargetControls();
  bindAudioControls();
}

function initAutonomyControls(): void {
  autonomy = new AutonomyController({
    clientId: liveClientId,
    request: body => sceneRequest('/api/sim/autonomy/decide', body),
    reset: async () => { await sceneRequest('/api/sim/autonomy/reset', { clientId: liveClientId }); },
    status: message => { document.querySelector('#autonomy-status')!.textContent = message; },
    vision: () => ego?.visualObservation() ?? null,
    enableVision: () => ego?.setEnabled(true),
    hold: reason => {
      if (activeCommandKind !== 'visual') return;
      cancelLiveGeneration(); motionPlaying = false; updateMotionReadout();
      reportCommand('interrupted', reason);
    },
    command: command => {
      const control: ControlCommand = { id: command.id, kind: command.kind, arguments: { ...command.arguments } };
      if (!rememberCommand(control.id)) return;
      void processCommand(control);
    },
  });
}

function bindTargetControls(): void {
  const queue = (path:string, args:object):void=>{
    void runSceneTask(async()=>{await sceneRequest(path,{...args,clientId:liveClientId});});
  };
  document.querySelector('#face-heading')!.addEventListener('click',()=>queue('/api/sim/face',{
    heading:Number(document.querySelector<HTMLInputElement>('#facing-angle')!.value)}));
  for(const [id,path] of [['face-target','face'],['move-target','approach'],['follow-target','follow']]) {
    document.querySelector(`#${id}`)!.addEventListener('click',()=>queue(`/api/sim/${path}`,{
      targetId:document.querySelector<HTMLSelectElement>('#target-select')!.value,
      distance:Number(document.querySelector<HTMLInputElement>('#follow-distance')!.value)}));
  }
  document.querySelector('#stop-follow')!.addEventListener('click',()=>queue('/api/sim/interrupt',{mode:'hold'}));
}

function bindAudioControls(): void {
  const microphone=new MicrophoneCapture();
  const start=document.querySelector<HTMLButtonElement>('#mic-start')!;
  const stop=document.querySelector<HTMLButtonElement>('#mic-stop')!;
  start.addEventListener('click',()=>{
    start.disabled=true;stop.disabled=false;
    void microphone.start().catch(error=>{
      start.disabled=false;stop.disabled=true;
      document.querySelector('#audio-status')!.textContent=error instanceof Error?error.message:'麦克风启动失败';
    });
  });
  stop.addEventListener('click',()=>{microphone.stop();start.disabled=false;stop.disabled=true;
    document.querySelector('#audio-status')!.textContent='麦克风已关闭';});
  const identityAction=async(modality:'face'|'speaker'|'delete'):Promise<void>=>{
    const name=document.querySelector<HTMLInputElement>('#identity-name')!.value;
    const sampleId=modality==='face'?ego?.faceSample():microphone.lastSpeaker?.sampleId;
    const body=modality==='delete'?{name}:{name,modality,sampleId,consent:document.querySelector<HTMLInputElement>('#identity-consent')!.checked};
    await perceptionRequest(`/api/identities/${modality==='delete'?'delete':'enroll'}`,body);
    document.querySelector('#identity-status')!.textContent=modality==='delete'?'本地登记已删除':'已保存本人同意的本地特征';
  };
  for(const [id,modality] of [['enroll-face','face'],['enroll-speaker','speaker'],['delete-identity','delete']] as const) {
    document.querySelector(`#${id}`)!.addEventListener('click',()=>{void identityAction(modality).catch(error=>{
      document.querySelector('#identity-status')!.textContent=error instanceof Error?error.message:'身份登记失败';});});
  }
  window.addEventListener('pagehide',()=>microphone.stop());
}

const timer = new THREE.Timer();
const power = new PowerMode(() => motionPlaying || !!activeLook || !!lumen?.active || preparingGestures.size > 0 || liveGenerationActive || !!follower?.active
  || !!autonomy?.active || !!character?.active || !!touch?.active || !!textContact?.active || !!document.querySelector<HTMLInputElement>('#touch-partner-moving')?.checked);

createModelCards();
createJointControls();
renderSnapshots();
renderTrajectoryEditor();
document.querySelector('#grid-toggle')!.addEventListener('change', (event) => { grid.visible = (event.currentTarget as HTMLInputElement).checked; });
document.querySelector('#axes-toggle')!.addEventListener('change', (event) => { axes.visible = (event.currentTarget as HTMLInputElement).checked; });
document.querySelectorAll<HTMLButtonElement>('.swatch').forEach((button) => button.addEventListener('click', () => {
  scene.background = new THREE.Color(button.dataset.bg!);
  document.querySelectorAll('.swatch').forEach((swatch) => swatch.classList.toggle('selected', swatch === button));
}));
document.querySelector('#frame-model')!.addEventListener('click', () => { if (currentVrm) frameAvatar(currentVrm); });
document.querySelector('#reset-joint')!.addEventListener('click', resetCurrentJoint);
document.querySelector('#reset-pose')!.addEventListener('click', resetAll);
document.querySelector('#reset-all')!.addEventListener('click', resetAll);
document.querySelector('#save-pose')!.addEventListener('click', () => {
  if (!currentVrm) return;
  snapshots.push({ name: `姿势 ${String(snapshots.length + 1).padStart(2, '0')}`, values: capturePose() });
  renderSnapshots();
});

document.querySelector<HTMLSelectElement>('#motion-select')!.addEventListener('change', (event) => {
  const entry = ardyMotions.find((motion) => motion.id === (event.currentTarget as HTMLSelectElement).value);
  if (entry) void loadMotion(entry, true);
});
document.querySelector('#play-motion')!.addEventListener('click', () => setMotionPlaying(livePriming ? false : !motionPlaying));
document.querySelector('#stop-motion')!.addEventListener('click', () => {
  if (sceneReady && houseEnabled) {
    autonomy?.suspend();
    void runSceneTask(async () => { await sceneRequest('/api/sim/interrupt', { clientId: liveClientId, mode: 'hold' }); });
    return;
  }
  reportCommand('interrupted', '网页停止播放');
  if (motionMode !== 'ardy') { stopWithAnimation(); return; }
  cancelLiveGeneration();
  motionPlaying = false;
  if (!houseEnabled) motionTime = 0;
  applyMotionFrame();
  updateMotionReadout();
});
document.querySelector<HTMLInputElement>('#motion-loop')!.addEventListener('change', (event) => {
  motionLoop = (event.currentTarget as HTMLInputElement).checked && motionLoopable;
});
document.querySelector<HTMLSelectElement>('#motion-speed')!.addEventListener('change', (event) => {
  motionSpeed = Number((event.currentTarget as HTMLSelectElement).value);
});
const motionScrub = document.querySelector<HTMLInputElement>('#motion-scrub')!;
motionScrub.addEventListener('pointerdown', () => { reportCommand('interrupted', '手动拖动时间轴'); cancelLiveGeneration(); motionScrubbing = true; motionPlaying = false; updateMotionReadout(); });
motionScrub.addEventListener('pointerup', () => { motionScrubbing = false; });
motionScrub.addEventListener('input', () => {
  if (!currentClip) return;
  reportCommand('interrupted', '手动调整时间轴');
  cancelLiveGeneration();
  motionTime = Number(motionScrub.value) / 1000 * currentClip.motion.duration;
  applyMotionFrame();
  updateMotionReadout();
});
document.querySelector('#generate-motion')!.addEventListener('click', () => { void generateArdyMotion(); });
document.querySelector('#generate-trajectory')!.addEventListener('click', () => { void generateTrajectoryMotion(); });
document.querySelector('#preview-route')!.addEventListener('click', () => { void planRoute(false); });
document.querySelector('#execute-route')!.addEventListener('click', () => { void planRoute(true); });
document.querySelector('#preview-plan')!.addEventListener('click', () => { void planTimeline(false); });
document.querySelector('#execute-plan')!.addEventListener('click', () => { void planTimeline(true); });
document.querySelector('#reset-house')!.addEventListener('click', () => {
  autonomy?.suspend();
  void runSceneTask(async () => { await sceneRequest('/api/sim/interrupt', { clientId: liveClientId, mode: 'reset' }); });
});
document.querySelector('#frame-house')!.addEventListener('click', () => { house?.frame(camera, controls); });
document.querySelector('#cutaway-toggle')!.addEventListener('change', event => { house?.setCutaway((event.target as HTMLInputElement).checked); });
document.querySelector('#collider-toggle')!.addEventListener('change', event => { if (house) house.colliders.visible = (event.target as HTMLInputElement).checked; });
document.querySelector('#house-toggle')!.addEventListener('change', event => {
  reportCommand('interrupted', '切换仿真模式');
  houseEnabled = (event.target as HTMLInputElement).checked;
  if (house) house.group.visible = houseEnabled;
  grid.visible = !houseEnabled;
  cancelLiveGeneration(); motionPlaying = false;
  if (houseEnabled) resetHouse();
});
document.querySelector('#add-trajectory-point')!.addEventListener('click', addTrajectoryPoint);
document.querySelector<HTMLDivElement>('#trajectory-rows')!.addEventListener('input', (event) => {
  const target = event.target;
  if (target instanceof HTMLInputElement) updateTrajectoryFromInput(target);
});
document.querySelector<HTMLDivElement>('#trajectory-rows')!.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const deleteButton = target.closest<HTMLButtonElement>('[data-delete]');
  if (!deleteButton || trajectoryPoints.length <= 2) return;
  trajectoryPoints.splice(Number(deleteButton.dataset.delete), 1);
  renderTrajectoryEditor();
});

document.querySelector('#motion-table-gesture')!.addEventListener('click', () => {
  const status = document.querySelector('#motion-table-result')!;
  try {
    if (!locomotion) throw new Error('动作表尚未就绪');
    if (activeCommand) throw new Error('请先暂停当前自主行为，再执行手势');
    const id = document.querySelector<HTMLSelectElement>('#motion-table-select')!.value;
    const entry = locomotion.table.document.entries.find(row => row.id === id);
    if (!entry?.prompt || entry.role !== 'action') throw new Error('请选择交互动作类别，移动片段请使用曲线路线');
    const seed = currentPresetSeed(), result = locomotion.tableAction(entry.prompt, seed, safePresetRoot);
    installPreset(result, seed); status.textContent = `完整手势 · ${result.seconds.toFixed(2)} 秒 · 全身姿态接缝`;
  } catch (error) { status.textContent = error instanceof Error ? error.message : '手势执行失败'; }
});

document.querySelector('#motion-table-test')!.addEventListener('click', () => {
  const status = document.querySelector('#motion-table-result')!;
  try {
    if (!locomotion) throw new Error('动作表尚未就绪');
    if (activeCommand) throw new Error('请先暂停当前自主行为，再试走曲线');
    const seed = currentPresetSeed(), clip = SmplClip.from(seed.motion);
    const origin = clip.rootAt(clip.frames-1, new THREE.Vector3());
    const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(new THREE.Quaternion().fromArray(clip.local, (clip.frames-1)*27*4));
    const yaw = Math.atan2(forward.x, forward.z);
    const path = Array.from({ length: 25 }, (_, i) => {
      const t = i/24*1.1, point = new THREE.Vector3(3*(1-Math.cos(t)), 0, 3*Math.sin(t));
      return point.applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw).add(origin).toArray();
    });
    const result = locomotion.compileTable('curve-preview', path, seed, safePresetRoot);
    installPreset(result, seed);
    status.textContent = `${result.table!.entries.length} 段连续执行 · 偏差 ${(result.table!.pathMaxError*100).toFixed(1)}cm · 检索 ${result.table!.searchMs.toFixed(0)}ms`;
  } catch (error) { status.textContent = error instanceof Error ? error.message : '动作表试走失败'; }
});

void (async () => {
  await loadModel(activeModel);
  await initArdyLibrary();
  await initHouse();
  actionLibrary = new ActionLibraryPanel(ardyService, async selection => {
    await sceneRequest('/api/sim/action', { ...selection, clientId: liveClientId });
  }, clips => {
    const saved = clips.map(clip => ({ id: clip.id, file: clip.id, name: `${clip.action} · ${clip.id.slice(-6)}`,
      dataset: '已校验播放库', fps: 20, frames: Math.round(clip.duration*20), duration: clip.duration,
      bytes: 0, sourceUrl: `${ardyService}/api/library/clips/${clip.id}` }));
    ardyMotions = [...ardyMotions.filter(entry => !entry.id.startsWith('clip_')), ...saved];
    renderMotionOptions();
  });
  try { await actionLibrary.refresh(); }
  catch (error) { document.querySelector('#library-status')!.textContent = error instanceof Error ? error.message : '动作库不可用'; }
})();
initPerceptionControls();
animate();

import * as THREE from 'three';
import type { PerceptionCadence } from './power';
import type { VRM } from '@pixiv/three-vrm';
import type { HouseScene } from './house';
import { perceptionRequest, PerceptionRequestError, type Face, type VisionResult, type VisualObservation } from './perception';
import { StereoCapture, type StereoFrame, type StereoResult } from './stereo';
import { VisualWorldMap } from './worldmap';
import { measureVrmRest } from './retarget';

type EgoContext = { scene: THREE.Scene; renderer: THREE.WebGLRenderer; avatar: THREE.Group;
  vrm: () => VRM | null; house: () => HouseScene | null; heading: () => number;
  cadence: () => PerceptionCadence;
  navigating: () => boolean;
  observe: (target: object) => Promise<object>; onDepth?: (depth: StereoResult, travelBearing: number) => void };
type ProcessedFrame = { frame: StereoFrame; depth: StereoResult };
type ObservationResult = { depth: StereoResult; vision: VisionResult | null;
  pipeline: { mode: string; elapsedMs: number; ageSeconds: number; queuedFrames: number; visionError?: string } };
type ObservationRequest = { clientId: string; frameId: number; observedAt: number; left: string; right: string;
  calibration: StereoFrame['calibration']; rig: StereoFrame['rig']; classes: string[]; includeVision: boolean };

export class EgoPerception {
  readonly camera = new THREE.PerspectiveCamera(78, 640/360, .06, 60);
  private captureRig = new StereoCapture();
  private enabled = false;
  private revision = 0;
  private clientId = `stereo-${crypto.randomUUID()}`;
  private controllers = new Set<AbortController>();
  private timers = new Set<number>();
  private faces: Face[] = [];
  private semantics: VisionResult | null = null;
  private semanticImage = '';
  private semanticPreview: HTMLImageElement | null = null;
  private semanticFrame: StereoFrame | null = null;
  private semanticViewHeading: number | undefined;
  private semanticPose: number[] | undefined;
  private semanticBodyPosition: number[] | undefined;
  private counters = { captures: 0, depthRequests: 0, semanticRequests: 0 };
  private processed: ProcessedFrame | null = null;
  private latestFrame: StereoFrame | null = null;
  private gazePitch: number | null = null;
  private gazeUpdated = 0;
  private groundLookUntil = 0;
  private backoff = { depth: 0, semantic: 0 };
  private retryAt = { depth: 0, semantic: 0 };
  private semanticRequestedAt = -Infinity;
  private observeUnavailableUntil = 0;
  private pipeline: ObservationResult['pipeline'] | null = null;
  private worldView: VisualWorldMap;

  constructor(private context: EgoContext) {
    this.worldView = new VisualWorldMap(document.querySelector<HTMLCanvasElement>('#world-cloud')!);
    window.addEventListener('pagehide', () => {
      this.setEnabled(false); this.revision++; this.stopRequests();
      void perceptionRequest('/api/stereo/release', { clientId: this.clientId }, undefined, true)
        .catch(error => console.error('退出页面时释放双目会话失败', error));
    });
    document.querySelector('#ego-toggle')!.addEventListener('change', event => {
      this.setEnabled((event.target as HTMLInputElement).checked);
    });
    document.querySelector('#reset-visual-map')!.addEventListener('click', () => {
      void this.resetMap().catch(error => this.status(error instanceof Error ? error.message : '地图重置失败'));
    });
    document.querySelector('#ego-fov')!.addEventListener('change', () => {
      void this.resetMap().catch(error => this.status(error instanceof Error ? error.message : '视野标定重置失败'));
    });
  }

  faceSample(): string {
    if (this.faces.length !== 1) throw new Error('需要恰好一张清晰人脸；请先开启第一人称检测');
    return this.faces[0].sampleId;
  }

  get sessionId(): string { return this.clientId; }

  navigationHeading(): number {
    this.updateCamera();
    const forward = this.camera.getWorldDirection(new THREE.Vector3());
    return Math.atan2(forward.x, forward.z);
  }

  diagnostics(): object | null {
    const depth = this.processed?.depth;
    if (!this.enabled || !depth) return null;
    return { frameId: depth.frameId, ageSeconds: Math.max(0, Date.now()/1000-depth.observedAt),
      mapping: depth.mapping, poseStatus: depth.pose.status, inliers: depth.pose.inliers,
      validRatio: depth.validRatio, inferenceMs: depth.inferenceMs,
      headBound: true, viewHeading: this.processed?.frame.viewHeading,
      semanticAgeSeconds: this.semantics ? Math.max(0, Date.now()/1000-this.semantics.observedAt) : null,
      pipeline: this.pipeline,
      people: this.semantics?.people?.length ?? 0, ...this.counters, cadence: this.context.cadence() };
  }

  touchPeople(): VisionResult | null {
    return this.enabled && this.semantics && Date.now()/1000-this.semantics.observedAt < 1.5 ? this.semantics : null;
  }

  touchImage(): string { return this.semanticImage; }

  targetFrame(): { image: string; frameId: number; observedAt: number } | null {
    const current = this.processed;
    if (!this.enabled || !current || Date.now()/1000-current.depth.observedAt >= 1.5) return null;
    return { image: current.frame.left, frameId: current.depth.frameId, observedAt: current.depth.observedAt };
  }

  visualObservation(): VisualObservation | null {
    const current = this.processed;
    const freshDepth = !!current && Date.now()/1000-current.depth.observedAt < 1;
    const frame = freshDepth ? current.frame : this.latestFrame;
    if (!this.enabled || !frame) return null;
    const depth = current?.depth, semantics = this.semantics;
    const fresh = semantics && Date.now()/1000-semantics.observedAt < 1.5;
    // Fresh local RGB remains actionable while depth is missing, delayed or throttled.
    // Never pair an old map/depth estimate with the new exposure as fresh evidence.
    return { objects: fresh ? semantics.objects : [], faces: fresh ? semantics.faces : [],
      width: 640, height: 360, inferenceMs: depth?.inferenceMs ?? 0, observedAt: frame.observedAt,
      semanticObservedAt: fresh ? semantics.observedAt : undefined, semanticReady: !!fresh,
      semanticFusion: fresh ? semantics.semanticFusion : undefined, people: fresh ? semantics.people : [],
      signature: frame.signature, viewHeading: frame.viewHeading, bodyHeading: frame.bodyHeading,
      semanticViewHeading: fresh ? this.semanticViewHeading : undefined,
      semanticPose: fresh ? this.semanticPose : undefined,
      semanticBodyPosition: fresh ? this.semanticBodyPosition : undefined,
      stereo: { method: depth?.method ?? 'unavailable', observedAt: depth?.observedAt ?? 0,
        validRatio: freshDepth ? depth!.validRatio : 0, sectors: freshDepth ? depth!.sectors : [],
        pose: depth?.pose ?? { status: 'initializing', position: [], cameraToWorld: [],
          inliers: 0, reprojectionError: null },
        mapping: freshDepth && depth!.mapping, navigationGrid: freshDepth ? depth!.navigationGrid : undefined,
        exitEvidence: freshDepth ? depth!.exitEvidence : undefined,
        groundInfo: freshDepth ? depth!.groundInfo : undefined } };
  }

  setEnabled(enabled: boolean): void {
    if (this.enabled === enabled) return;
    this.enabled = enabled; this.revision++;
    document.querySelector<HTMLInputElement>('#ego-toggle')!.checked = enabled;
    this.stopRequests();
    if (enabled) {
      this.previewTick(this.revision);
      void this.depthTick(this.revision);
    } else {
      this.processed = null; this.latestFrame = null; this.faces = []; this.semantics = null;
      this.semanticPreview = null; this.semanticFrame = null;
      this.status('双目感知已关闭');
    }
  }

  private stopRequests(): void {
    for (const controller of this.controllers) controller.abort();
    this.controllers.clear();
    for (const timer of this.timers) clearTimeout(timer);
    this.timers.clear();
  }

  private async resetMap(): Promise<void> {
    const resume = this.enabled;
    this.setEnabled(false);
    const revision = this.revision;
    try {
      await perceptionRequest('/api/stereo/reset', { clientId: this.clientId });
      this.processed = null; this.semantics = null; this.worldView.resetView();
      const status = document.querySelector<HTMLElement>('#world-status')!;
      status.textContent = '视觉地图已重置 · 等待新原点'; delete status.dataset.state;
    } finally { if (revision === this.revision) this.setEnabled(resume); }
  }

  resetOrigin(): void {
    void this.resetMap().catch(error => this.status(error instanceof Error ? error.message : '地图重置失败'));
  }

  private schedule(callback: () => void, delay: number, revision: number): void {
    if (!this.enabled || revision !== this.revision) return;
    const timer = window.setTimeout(() => { this.timers.delete(timer); callback(); }, delay);
    this.timers.add(timer);
  }

  private scheduleCadence(callback: () => void, kind: keyof PerceptionCadence, started: number, revision: number): void {
    const poll = (): void => {
      const retry = kind === 'preview' ? 0 : this.retryAt[kind]-performance.now();
      const remaining = Math.max(retry, this.context.cadence()[kind]-(performance.now()-started));
      if (remaining <= 0) callback();
      else this.schedule(poll, Math.min(remaining, 100), revision);
    };
    this.schedule(poll, 0, revision);
  }

  private updateCamera(): void {
    const vrm = this.context.vrm();
    const head = vrm?.humanoid.getRawBoneNode('head');
    if (!vrm || !head) throw new Error('请先加载带头部骨骼的 VRM 模型');
    const fov = THREE.MathUtils.clamp(Number(document.querySelector<HTMLInputElement>('#ego-fov')!.value) || 78, 60, 120);
    if (fov !== this.camera.fov) { this.camera.fov = fov; this.camera.updateProjectionMatrix(); }
    this.context.avatar.updateWorldMatrix(true, true);
    head.getWorldPosition(this.camera.position);
    const normalized = vrm.humanoid.getNormalizedBoneNode('head');
    const rest = measureVrmRest(vrm).get('head');
    if (!normalized || !rest) throw new Error('缺少头部朝向标定');
    normalized.getWorldQuaternion(this.camera.quaternion).multiply(rest.clone().invert());
    this.camera.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI));
    const pitch = THREE.MathUtils.degToRad(this.navigationPitch());
    this.camera.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -pitch));
    this.camera.position.addScaledVector(this.camera.getWorldDirection(new THREE.Vector3()), .10);
    this.camera.updateMatrixWorld(true);
  }

  private navigationPitch(): number {
    const baseline = THREE.MathUtils.clamp(Number(document.querySelector<HTMLInputElement>('#ego-pitch')!.value) || 0, -45, 60);
    const now = performance.now(), depth = this.processed?.depth;
    const navigating = this.context.navigating();
    if (navigating && depth && (!depth.mapping || depth.validRatio < .08 || !depth.groundInfo)) this.groundLookUntil = now+2000;
    const target = navigating && now < this.groundLookUntil ? Math.max(baseline, 22) : baseline;
    const step = Math.min(.25, Math.max(0, (now-this.gazeUpdated)/1000))*12;
    this.gazePitch = this.gazePitch === null ? baseline
      : this.gazePitch+THREE.MathUtils.clamp(target-this.gazePitch, -step, step);
    this.gazeUpdated = now;
    return this.gazePitch;
  }

  private capture(): StereoFrame {
    const { scene, renderer, avatar } = this.context;
    this.updateCamera();
    const prior = renderer.getRenderTarget(), visible = avatar.visible;
    const house = this.context.house();
    const cutaway = document.querySelector<HTMLInputElement>('#cutaway-toggle')!.checked;
    const overlays = house ? [house.routeGroup, house.colliders,
      ...house.group.children.filter(child => child instanceof THREE.Sprite)] : [];
    const visibility = overlays.map(object => object.visible);
    avatar.visible = false; overlays.forEach(object => { object.visible = false; }); house?.setCutaway(false);
    try {
      const frame = this.captureRig.capture(scene, renderer, this.camera);
      const forward = this.camera.getWorldDirection(new THREE.Vector3());
      frame.viewHeading = THREE.MathUtils.radToDeg(Math.atan2(forward.x, forward.z));
      frame.bodyHeading = THREE.MathUtils.radToDeg(this.context.heading());
      const leftEye = this.camera.matrixWorld.clone().multiply(new THREE.Matrix4().makeTranslation(-this.captureRig.baseline/2, 0, 0));
      const optical = leftEye.multiply(new THREE.Matrix4().makeScale(1, -1, -1));
      const bodyInOptical = avatar.getWorldPosition(new THREE.Vector3()).applyMatrix4(optical.clone().invert()).toArray();
      frame.rig = { source: 'ego-kinematics', opticalToControl: optical.clone().transpose().toArray(), bodyInOptical };
      return frame;
    } finally {
      renderer.setRenderTarget(prior); avatar.visible = visible; house?.setCutaway(cutaway);
      overlays.forEach((object, index) => { object.visible = visibility[index]; });
    }
  }

  private async request<T>(path: string, body: object, timeoutMs: number): Promise<T> {
    const controller = new AbortController(); this.controllers.add(controller);
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    try { return await perceptionRequest<T>(path, body, controller.signal); }
    finally { clearTimeout(timeout); this.controllers.delete(controller); }
  }

  private async observeFrame(body: ObservationRequest): Promise<ObservationResult> {
    if (performance.now() >= this.observeUnavailableUntil) {
      try {
        const result = await this.request<ObservationResult>('/api/observe', body, 4000);
        this.observeUnavailableUntil = 0;
        return result;
      } catch (error) {
        if (!(error instanceof PerceptionRequestError) || error.status !== 404) throw error;
        this.observeUnavailableUntil = performance.now()+30000;
      }
    }
    const started = performance.now();
    const depth = await this.request<StereoResult>('/api/stereo', body, 2500);
    let vision: VisionResult | null = null, visionError: string | undefined;
    if (body.includeVision) {
      try {
        vision = await this.request<VisionResult>('/api/vision', {
          clientId: body.clientId, frameId: body.frameId, observedAt: body.observedAt,
          image: body.left, classes: body.classes, pose: true,
        }, 5000);
      } catch (error) {
        if (!(error instanceof Error)) throw error;
        visionError = error.message;
      }
    }
    return { depth, vision, pipeline: { mode: 'legacy-sequential',
      elapsedMs: Math.round((performance.now()-started)*100)/100,
      ageSeconds: Math.max(0, Date.now()/1000-body.observedAt), queuedFrames: 0,
      ...(visionError ? { visionError } : {}) } };
  }

  private async depthTick(revision: number): Promise<void> {
    const started = performance.now();
    try {
      let frame = this.latestFrame;
      // Preview and depth have separate idle timers. Never process the previous
      // preview cycle just because the depth timer ran a moment before preview.
      if (!frame || Date.now()/1000-frame.observedAt > .5) {
        frame = this.capture(); this.latestFrame = frame; this.counters.captures++;
      }
      if (!frame || frame.frameId === this.processed?.frame.frameId) return;
      this.counters.depthRequests++;
      const includeVision = started-this.semanticRequestedAt >= this.context.cadence().semantic;
      if (includeVision) { this.semanticRequestedAt = started; this.counters.semanticRequests++; }
      const classes = document.querySelector<HTMLInputElement>('#vision-classes')!.value.split(',').map(s => s.trim()).filter(Boolean);
      const observation = await this.observeFrame({
        clientId: this.clientId, frameId: frame.frameId, observedAt: frame.observedAt,
        left: frame.left, right: frame.right, calibration: frame.calibration,
        rig: frame.rig, classes, includeVision,
      });
      if (revision !== this.revision || !this.enabled) return;
      const result = observation.depth; this.pipeline = observation.pipeline;
      const map = this.processed?.depth.map;
      if (map && map.mapId === result.map.mapId && map.revision > result.map.revision) result.map = map;
      this.backoff.depth = 0; this.retryAt.depth = 0;
      this.processed = { frame, depth: result };
      this.drawDepth(result); this.worldView.update(result);
      if (observation.vision) await this.acceptSemantics(observation.vision, { frame, depth: result }, revision);
      else if (observation.pipeline.visionError) document.querySelector('#vision-status')!.textContent = observation.pipeline.visionError;
      const bearing = (frame.viewHeading ?? 0)-(frame.bodyHeading ?? THREE.MathUtils.radToDeg(this.context.heading()));
      this.context.onDepth?.(result, THREE.MathUtils.radToDeg(Math.atan2(Math.sin(bearing*Math.PI/180), Math.cos(bearing*Math.PI/180))));
    } catch (error) {
      if (revision === this.revision && this.enabled) {
        this.deferBusy('depth', error);
        this.status(error instanceof Error ? error.message : '双目测距失败');
        document.querySelector('#world-status')!.textContent = '观测中断 · 点云保留为历史记忆 · 暂停新点融合';
      }
    } finally {
      this.scheduleCadence(() => { void this.depthTick(revision); }, 'depth', started, revision);
    }
  }

  private async acceptSemantics(result: VisionResult, current: ProcessedFrame, revision: number): Promise<void> {
      const { frame } = current;
      if (result.frameId !== frame.frameId || result.observedAt !== frame.observedAt)
        throw new Error('语义与双目曝光帧不一致');
      const preview = new Image(); preview.src = frame.left;
      await preview.decode();
      if (revision !== this.revision || !this.enabled) return;
      this.backoff.semantic = 0; this.retryAt.semantic = 0;
      this.semanticViewHeading = frame.viewHeading;
      this.semanticPose = current.depth.mapping ? [...current.depth.pose.cameraToWorld] : undefined;
      this.semanticBodyPosition = current.depth.mapping ? current.depth.pose.bodyPosition : undefined;
      this.semantics = result; this.faces = result.faces;
      this.semanticImage = frame.left;
      this.semanticPreview = preview; this.semanticFrame = frame;
      this.drawPreview();
      const fusion = result.semanticFusion;
      if (fusion?.map && this.processed && fusion.map.mapId === this.processed.depth.map.mapId
        && fusion.map.revision >= this.processed.depth.map.revision) {
        this.processed.depth.map = fusion.map;
        this.worldView.updateMap(fusion.map);
        this.drawMapStatus(this.processed.depth);
      }
      if (fusion) delete fusion.map;
      const detail = fusion?.fused ? `${fusion.associations?.length ?? 0} 个深度表面` : fusion?.reason ?? '未融合';
      document.querySelector('#vision-status')!.textContent = `${result.objects.length} 个物体 / ${result.faces.length} 张人脸 · ${result.inferenceMs.toFixed(0)} ms · ${detail}`;
  }

  /** Catch values are untrusted; only typed HTTP capacity errors change scheduling. */
  private deferBusy(kind: 'depth' | 'semantic', error: unknown): void {
    if (!(error instanceof PerceptionRequestError) || ![429, 503].includes(error.status)) return;
    this.backoff[kind] = Math.min(800, Math.max(120, this.backoff[kind]*1.7));
    this.retryAt[kind] = performance.now()+this.backoff[kind]+Math.random()*80;
  }

  private drawPair(): void {
    for (const [id, source] of [['stereo-left', this.captureRig.left], ['stereo-right', this.captureRig.right]] as const) {
      document.querySelector<HTMLCanvasElement>(`#${id}`)!.getContext('2d')!.drawImage(source, 0, 0, 320, 180);
    }
  }

  /** Local RGB remains live when either perception service is busy or offline. */
  private previewTick(revision: number): void {
    if (!this.enabled || revision !== this.revision) return;
    const started = performance.now();
    try {
      this.latestFrame = this.capture(); this.counters.captures++;
      this.drawPair(); this.drawPreview();
    } catch (error) { this.status(error instanceof Error ? error.message : '本地视角渲染失败'); }
    this.scheduleCadence(() => this.previewTick(revision), 'preview', started, revision);
  }

  private drawPreview(): void {
    const canvas = document.querySelector<HTMLCanvasElement>('#ego-preview')!;
    const ctx = canvas.getContext('2d')!;
    const semantic = this.semantics && this.semanticPreview && Date.now()/1000-this.semantics.observedAt < 1.5;
    const frame = semantic ? this.semanticFrame : this.latestFrame;
    ctx.drawImage(semantic ? this.semanticPreview! : this.captureRig.left, 0, 0, canvas.width, canvas.height);
    canvas.dataset.frameId = String(frame?.frameId ?? 0);
    canvas.dataset.viewHeading = String(frame?.viewHeading ?? 0);
    canvas.dataset.cameraQuaternion = JSON.stringify(frame?.cameraQuaternion ?? this.camera.quaternion.toArray());
    canvas.dataset.semanticFrameId = semantic ? String(this.semantics!.frameId ?? 0) : '';
    if (semantic) this.drawSemantics(this.semantics!);
  }

  private drawDepth(result: StereoResult): void {
    document.querySelector<HTMLImageElement>('#depth-preview')!.src = result.preview;
    const front = result.sectors.find(sector => sector.angle === 0);
    const clearance = front?.clearance == null ? '未知' : `${front.clearance.toFixed(2)} m`;
    const status = document.querySelector<HTMLElement>('#depth-status')!;
    const completion = result.surfaceCompletion?.inferredPixels ?? 0;
    status.textContent = `前方 ${clearance} · 有效深度 ${(result.validRatio*100).toFixed(0)}% · 补全 ${completion} 像素（暗色） · ${result.inferenceMs.toFixed(0)} ms`;
    status.dataset.result = JSON.stringify({ ...result, map: undefined, preview: undefined, depthGrid: undefined });
    this.drawMapStatus(result);
  }

  private drawMapStatus(result: StereoResult): void {
    const map = document.querySelector<HTMLElement>('#world-status')!;
    const ground = result.groundInfo;
    const groundText = ground?.frame === 'gravity-aligned-y-up'
      ? `地面锁定 ${ground.tiltDeg.toFixed(1)}°`
      : ground ? '地面参考有限' : '地面未知';
    const tilt = result.pose.tiltCorrectionDeg ?? 0;
    const surfaces = result.map.surfaces?.length ?? 0, volumes = result.map.volumes?.length ?? 0;
    map.textContent = `${result.pose.status} · 位置 [${result.pose.position.map(v => v.toFixed(2)).join(', ')}] m · ${result.map.totalPoints} 点（推断 ${result.map.inferredPoints ?? 0}） · ${surfaces} 面 / ${volumes} 体 · ${groundText} · 姿态校正 ${tilt.toFixed(1)}°`;
    map.dataset.state = JSON.stringify({ pose: result.pose, revision: result.map.revision,
      totalPoints: result.map.totalPoints, semanticPoints: result.map.semantics.filter(Boolean).length,
      inferredPoints: result.map.inferredPoints ?? 0, surfaces, volumes,
      geometryRevision: result.map.geometryRevision, geometry: result.map.geometry,
      objects: result.map.objects, clientId: this.clientId });
  }

  private drawSemantics(result: VisionResult): void {
    const ctx = document.querySelector<HTMLCanvasElement>('#ego-preview')!.getContext('2d')!;
    ctx.font = '16px sans-serif'; ctx.lineWidth = 2;
    for (const object of [...result.objects, ...result.faces.map(face => ({ ...face, label: face.identity ?? face.anonymousId }))]) {
      const [x0, y0, x1, y1] = object.box;
      ctx.strokeStyle = '#9ce6af'; ctx.fillStyle = '#9ce6af';
      ctx.strokeRect(x0, y0, x1-x0, y1-y0); ctx.fillText(object.label, x0, Math.max(16, y0-4));
    }
    ctx.strokeStyle = '#ffcb69'; ctx.lineWidth = 2;
    for (const person of result.people ?? []) for (const point of person.keypoints) {
      if (point.confidence < .4) continue;
      const [x, y] = point.pixel;
      ctx.fillStyle = point.valid ? '#ffcb69' : '#ff7b7b'; ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI*2); ctx.fill();
      if (point.name.includes('wrist')) ctx.fillText(`${point.name} ${point.distance?.toFixed(2) ?? '?'}m`, x+4, y);
    }
    for (const surface of result.semanticFusion?.associations ?? []) {
      if (surface.boundary.length < 3) continue;
      ctx.beginPath();
      surface.boundary.forEach(([x, y], index) => index === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
      ctx.closePath(); ctx.stroke();
    }
  }

  private status(message: string): void { document.querySelector('#depth-status')!.textContent = message; }
}

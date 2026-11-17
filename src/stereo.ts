import * as THREE from 'three';

export type DepthSector = { angle: number; clearance: number | null; confidence: number; obstacleDistance?: number | null };
export type VisualPose = { status: 'initializing' | 'initialized' | 'tracking' | 'relocalized' | 'lost';
  position: number[]; cameraToWorld: number[]; inliers: number; reprojectionError: number | null;
  bodyPosition?: number[];
  gravityAligned?: boolean; tiltCorrectionDeg?: number };
export type SemanticObject = { id: string; label: string; center: number[]; bounds: number[][];
  confidence: number; observations: number; ageSeconds: number; visible: boolean };
export type SurfaceSegment = { id: string; volumeId: string | null; normal: number[]; centroid: number[];
  bounds: number[][]; vertices: number[]; indices: number[]; color: number; meanColor: number[]; semantic: number;
  label: string; confidence: number; measuredRatio: number; voxelCount: number; continuous: boolean };
export type VolumeSegment = { id: string; label: string; semantic: number; semanticConfidence: number;
  centroid: number[]; bounds: number[][]; voxelCount: number; measuredRatio: number;
  surfaceIds: string[]; instanceId: string | null; lastSeen: number; geometry: string };
export type WorldCloud = { mapId?: string; revision: number; totalPoints: number; positions: number[]; colors: number[];
  semantics: number[]; labels: string[]; objects: SemanticObject[]; surfaces?: SurfaceSegment[];
  volumes?: VolumeSegment[]; geometryRevision?: number; volumeIds?: (string | null)[];
  geometry?: { surfaceCount: number; volumeCount: number; rebuildMs: number; usage: string };
  inferred?: boolean[]; inferredPoints?: number };
export type ExitEvidence = { source: string; frame: string; observedAt: number; revision: number;
  candidates: { position: number[]; normal: number[]; width: number; confidence: number;
    observedFrames: number; throughSupport: number; uncertaintyMeters: number; kind: string; state: string }[];
  sealed: boolean; inferenceMs: number };
export type StereoResult = { frameId: number; observedAt: number; method: string; validRatio: number;
  surfaceCompletion?: { method: string; planes: number; inferredPixels: number; inferenceMs: number; usage: string };
  exitEvidence?: ExitEvidence | null;
  navigationGrid?: { resolution: number; free: number[][]; occupied: number[][]; frame: string; source: string } | null;
  sectors: DepthSector[]; preview: string; pose: VisualPose; map: WorldCloud; trajectory: number[][];
  depthGrid: { width: number; height: number; meters: number[] }; inferenceMs: number; mapping: boolean;
  groundPlane?: number[] | null;
  groundInfo?: { normal: number[]; offset: number; support: number; residual: number; tiltDeg: number; frame: string } | null };
export type StereoFrame = { frameId: number; observedAt: number; left: string; right: string;
  viewHeading?: number; bodyHeading?: number;
  cameraQuaternion?: number[];
  rig?: { source: 'ego-kinematics'; opticalToControl: number[]; bodyInOptical?: number[] };
  signature: number[];
  calibration: { width: number; height: number; fx: number; fy: number; cx: number; cy: number; baseline: number } };

/** Parallel optical axes; only RGB pixels and fixed rig calibration leave the renderer. */
export class StereoCapture {
  readonly width = 640;
  readonly height = 360;
  readonly baseline = .12;
  readonly left = document.createElement('canvas');
  readonly right = document.createElement('canvas');
  private target = new THREE.WebGLRenderTarget(this.width, this.height);
  private pixels = new Uint8Array(this.width*this.height*4);
  private flipped = new Uint8ClampedArray(this.pixels.length);
  private frameId = 0;

  constructor() {
    for (const canvas of [this.left, this.right]) {
      canvas.width = this.width; canvas.height = this.height;
    }
  }

  capture(scene: THREE.Scene, renderer: THREE.WebGLRenderer, center: THREE.PerspectiveCamera): StereoFrame {
    const camera = center.clone();
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(center.quaternion);
    camera.position.addScaledVector(right, -this.baseline/2);
    camera.updateMatrixWorld(true);
    this.render(scene, renderer, camera, this.left);
    camera.position.addScaledVector(right, this.baseline);
    camera.updateMatrixWorld(true);
    this.render(scene, renderer, camera, this.right);
    const focal = this.height/(2*Math.tan(THREE.MathUtils.degToRad(center.fov)/2));
    return { frameId: ++this.frameId, observedAt: Date.now()/1000, cameraQuaternion: center.quaternion.toArray(),
      left: this.left.toDataURL('image/jpeg', .92), right: this.right.toDataURL('image/jpeg', .92),
      signature: this.signature(),
      calibration: { width: this.width, height: this.height, fx: focal, fy: focal,
        cx: this.width/2, cy: this.height/2, baseline: this.baseline } };
  }

  private render(scene: THREE.Scene, renderer: THREE.WebGLRenderer, camera: THREE.PerspectiveCamera,
    canvas: HTMLCanvasElement): void {
    renderer.setRenderTarget(this.target); renderer.render(scene, camera);
    renderer.readRenderTargetPixels(this.target, 0, 0, this.width, this.height, this.pixels);
    const rowSize = this.width*4;
    for (let row = 0; row < this.height; row++) {
      this.flipped.set(this.pixels.subarray(row*rowSize, (row+1)*rowSize), (this.height-1-row)*rowSize);
    }
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('无法创建双目图像画布');
    context.putImageData(new ImageData(this.flipped, this.width, this.height), 0, 0);
  }

  private signature(): number[] {
    const context = this.left.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('无法读取左眼像素特征');
    const pixels = context.getImageData(0, 0, this.width, this.height).data;
    const values: number[] = [];
    for (let row = 0; row < this.height; row += 12) {
      for (let column = 0; column < this.width; column += 12) {
        const index = (row*this.width+column)*4;
        values.push((pixels[index]*.299+pixels[index+1]*.587+pixels[index+2]*.114)/255);
      }
    }
    return values;
  }
}

export function safeStereo(stereo: StereoResult | null, now = Date.now()/1000, bearing = 0, knownFloor = false): string | null {
  if (!stereo || now-stereo.observedAt > 1 || now < stereo.observedAt-.1) return '双目观测过期';
  if (!stereo.mapping) return '视觉定位丢失';
  if ('groundInfo' in stereo && stereo.groundInfo?.frame !== 'gravity-aligned-y-up') return '重力地面未知';
  const ground = stereo.groundInfo;
  if (ground && (ground.support < 40 || ground.tiltDeg > 12 || ground.residual > .04)) return '地板支撑或坡度证据不足';
  const front = stereo.sectors.reduce<DepthSector | undefined>((best, sector) =>
    !best || Math.abs(sector.angle-bearing) < Math.abs(best.angle-bearing) ? sector : best, undefined);
  if (!front || Math.abs(front.angle-bearing) > 8) return '行进方向超出当前双目视野';
  if (!front || front.clearance === null || front.confidence < .25) return '前方深度未知';
  // A mapped route can leave the current image's floor footprint. This is
  // distinct from a measured obstacle, which always retains the stop margin.
  const clearance = knownFloor && front.obstacleDistance !== undefined ? front.obstacleDistance ?? Infinity : front.clearance;
  if (clearance < .85) return `前方净空仅 ${clearance.toFixed(2)} 米`;
  return null;
}

import type { StereoResult, WorldCloud } from './stereo';
export { EgoPerception } from './ego_stereo';

export type ObservedTarget = { id: string; label: string; source: string; position?: number[];
  bearing?: number; observedAt: number; ageSeconds: number; ttl: number; stale: boolean };
type Identity = { sampleId: string; identity: string | null; anonymousId: string; similarity: number };
type Detection = { label: string; confidence: number; box: number[] };
export type Face = Identity & { box: number[]; confidence: number; landmarks?: number[][] };
export type SemanticAssociation = { id: string; label: string; points: number; depth: number;
  measuredPoints?: number; inferredPoints?: number;
  positionSigmaMeters?: number;
  volumeIds?: string[]; surfaceIds?: string[];
  surfacePixel?: number[];
  box: number[]; boundary: number[][]; worldPosition?: number[]; confidence?: number };
export type BodyKeypoint = { name: string; pixel: number[]; confidence: number; valid: boolean;
  opticalPosition?: number[]; worldPosition?: number[]; distance?: number; depth?: number;
  sigmaMeters?: number; reason?: string };
export type VisualPerson = { id: string; frameId: number; box: number[]; confidence: number; keypoints: BodyKeypoint[];
  opticalPosition: number[] | null; distance: number | null; observedAt: number; source: string };
export type VisionResult = { objects: Detection[]; faces: Face[]; width: number; height: number;
  frameId?: number;
  inferenceMs: number; observedAt: number; people?: VisualPerson[];
  semanticFusion?: { fused: boolean; reason?: string; frameId?: number; associations?: SemanticAssociation[];
    mapRevision?: number; map?: WorldCloud } };
export type VisualObservation = VisionResult & { signature: number[];
  semanticReady?: boolean;
  viewHeading?: number;
  bodyHeading?: number; semanticViewHeading?: number;
  semanticPose?: number[];
  semanticBodyPosition?: number[];
  semanticObservedAt?: number;
  stereo: Pick<StereoResult, 'method' | 'observedAt' | 'validRatio' | 'sectors' | 'pose' | 'mapping' | 'navigationGrid' | 'groundInfo' | 'exitEvidence'> };
export type AudioResult = { text: string; segments: { start: number; end: number; text: string }[];
  speaker: Identity | null; direction: { available: boolean; azimuth?: number; reason: string };
  inferenceMs?: number };
export const perceptionPanel = `
  <section class="perception-panel">
    <div class="section-head"><div><p class="eyebrow">EGO / PERCEPTION</p><h2>第一人称感知</h2></div></div>
    <label class="setting-row"><span>双目视觉 / YOLO-World</span><input id="ego-toggle" type="checkbox"><i class="switch"></i></label>
    <canvas id="ego-preview" width="640" height="360" aria-label="VRM 第一人称相机"></canvas>
    <label class="field-label" for="ego-pitch">视线俯角（度，负值抬头）</label>
    <input id="ego-pitch" class="scene-input" type="number" min="-45" max="60" step="5" value="15">
    <label class="field-label" for="ego-fov">垂直视野角（近距离可扩大，改变后重建视觉地图）</label>
    <input id="ego-fov" class="scene-input" type="number" min="60" max="120" step="5" value="78">
    <label class="field-label" for="vision-classes">检测类别（英文逗号分隔）</label>
    <input id="vision-classes" class="scene-input" value="person,face,chair,sofa,couch,table,bed,door,refrigerator,floor,wall,window,cabinet,remote control,computer monitor,keyboard,computer mouse,sandwich,cup,water dispenser,bottle">
    <p id="vision-status" class="ardy-status">语义检测 · 640×360 · 最多 2 Hz</p>
    <details class="stereo-panel" open><summary>双目深度 / 视觉世界模型</summary>
      <div class="stereo-pair"><figure><canvas id="stereo-left" width="320" height="180"></canvas><figcaption>左眼</figcaption></figure><figure><canvas id="stereo-right" width="320" height="180"></canvas><figcaption>右眼 · 基线 12 cm</figcaption></figure></div>
      <img id="depth-preview" width="320" height="180" alt="双目深度与图像边缘，黑色表示未知">
      <p id="depth-status" class="ardy-status">等待同步双目图像 · 目标 5 Hz</p>
      <canvas id="world-cloud" width="320" height="230" aria-label="视觉重建表面、点云与语义体积，可拖动旋转"></canvas>
      <p id="world-status" class="ardy-status">视觉定位未初始化 · 不读取仿真坐标</p>
      <ol id="world-objects" class="plan-timeline"></ol>
      <button id="reset-visual-map" class="secondary-button">重置视觉地图 / 定位原点</button>
    </details>
    <div class="scene-buttons"><button id="mic-start" class="secondary-button">开启麦克风</button><button id="mic-stop" class="secondary-button" disabled>停止录音</button></div>
    <p id="audio-status" class="ardy-status">本机麦克风 → ASR / 匿名声纹；单声道无方向</p>
    <ol id="speech-transcript" class="plan-timeline" aria-live="polite"></ol>
    <details><summary>本地身份登记（需本人同意）</summary>
      <input id="identity-name" class="scene-input" placeholder="登记名称" aria-label="登记名称">
      <label class="setting-row"><span>样本本人已同意登记</span><input id="identity-consent" type="checkbox"></label>
      <div class="scene-buttons"><button id="enroll-face" class="secondary-button">登记当前人脸</button><button id="enroll-speaker" class="secondary-button">登记当前说话人</button></div>
      <button id="delete-identity" class="secondary-button">删除该名称登记</button>
      <p id="identity-status" class="ardy-status">只保存本地特征，不保存原始录音/人脸图像；多人同时说话暂不分离。</p>
    </details>
  </section>`;

export class PerceptionRequestError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

export async function perceptionRequest<T>(path: string, body?: object, signal?: AbortSignal, keepalive = false): Promise<T> {
  const response = await fetch(`${import.meta.env.VITE_MOTION_PERCEPTION_URL || 'http://127.0.0.1:8200'}${path}`, {
    ...(body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}), signal, keepalive,
  });
  const result = await response.json() as T & { error?: string };
  if (!response.ok || result.error) throw new PerceptionRequestError(result.error ?? `感知服务 HTTP ${response.status}`, response.status);
  return result;
}

export class MicrophoneCapture {
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private worklet: AudioWorkletNode | null = null;
  private busy = false;
  private revision = 0;
  lastSpeaker: Identity | null = null;

  async start(): Promise<void> {
    this.stop(); const revision = this.revision;
    const stream = await navigator.mediaDevices.getUserMedia({ audio:{ echoCancellation:false, noiseSuppression:false, autoGainControl:false, channelCount:1 } });
    if (revision !== this.revision) { stream.getTracks().forEach(track=>track.stop()); return; }
    this.stream = stream;
    try {
      const context = new AudioContext(); this.context = context;
      await context.resume();
      const code = `class Capture extends AudioWorkletProcessor {
        constructor(){super();this.data=new Float32Array(Math.round(sampleRate*4));this.offset=0;}
        process(inputs){const input=inputs[0]?.[0];if(input)for(let i=0;i<input.length;i++){
          this.data[this.offset++]=input[i];if(this.offset===this.data.length){this.port.postMessage(this.data);this.offset=0;}}
          return true;}}
        registerProcessor('motion-capture',Capture);`;
      const url = URL.createObjectURL(new Blob([code],{type:'application/javascript'}));
      try { await context.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
      if (revision !== this.revision) return;
      const worklet = new AudioWorkletNode(context,'motion-capture'); this.worklet = worklet;
      worklet.port.onmessage = (event: MessageEvent<Float32Array>) => { void this.process(event.data,context.sampleRate,revision); };
      const gain = context.createGain(); gain.gain.value=0;
      context.createMediaStreamSource(stream).connect(worklet); worklet.connect(gain).connect(context.destination);
      document.querySelector('#audio-status')!.textContent = '录音中 · 每4秒识别 · 单声道方向不可用';
    } catch (error) { this.stop(); throw error; }
  }

  stop(): void {
    this.revision++; this.stream?.getTracks().forEach(track=>track.stop()); this.stream=null;
    this.worklet?.disconnect(); this.worklet=null;
    if (this.context) void this.context.close().catch(error=>console.error('关闭麦克风失败',error));
    this.context=null;
  }

  private async process(samples: Float32Array, rate: number, revision: number): Promise<void> {
    if (this.busy || revision !== this.revision) return;
    const rms = Math.sqrt(samples.reduce((sum,sample)=>sum+sample*sample,0)/samples.length);
    if (rms < .003) return;
    this.busy=true;
    try {
      const pcm = new Int16Array(samples.length);
      samples.forEach((value,index)=>{pcm[index]=Math.max(-32768,Math.min(32767,Math.round(value*32768)));});
      let binary=''; const bytes = new Uint8Array(pcm.buffer);
      for (let i=0;i<bytes.length;i+=16384) binary+=String.fromCharCode(...bytes.subarray(i,i+16384));
      const result = await perceptionRequest<AudioResult>('/api/audio',{pcm16:btoa(binary),sampleRate:rate,channels:1,
        observedAt:Date.now()/1000,language:'zh'},AbortSignal.timeout(30000));
      if (revision !== this.revision) return;
      this.lastSpeaker=result.speaker;
      document.querySelector('#audio-status')!.textContent = `${result.speaker?.identity ?? result.speaker?.anonymousId ?? '无有效语音'} · ${result.direction.reason} · ${(result.inferenceMs ?? 0).toFixed(0)} ms`;
      const list=document.querySelector('#speech-transcript')!;
      for(const segment of result.segments) { const item=document.createElement('li'); item.textContent=`${segment.start.toFixed(1)}–${segment.end.toFixed(1)}s ${segment.text}`; list.appendChild(item); }
      while(list.children.length>12) list.firstElementChild?.remove();
    } catch (error) { if(revision===this.revision) document.querySelector('#audio-status')!.textContent=error instanceof Error?error.message:'音频失败'; }
    finally { this.busy=false; }
  }
}

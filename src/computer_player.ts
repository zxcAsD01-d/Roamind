import * as THREE from 'three';
import type { VRM } from '@pixiv/three-vrm';
import { contactBone } from './contact_rig';
import { measureVrmRest } from './retarget';
import { ComputerStation, matchComputer, type ComputerState } from './computer';
import { applyPose, capturePose, palmPosition, point, reachPalm, setWorldRotation, type Pose } from './interaction_pose';

export class ComputerPlayCancelled extends Error {}
type Session = { elapsed: number; duration: number; resolve: () => void; reject: (error: Error) => void };

/** Owns hand input only; sitting, standing and transitions remain owned by InteractionBody. */
export class ComputerPlayer {
  private station: ComputerStation | null = null;
  private session: Session | null = null;
  private targets: THREE.Vector3[] = [];
  private forward = new THREE.Vector3();
  constructor(private vrm: () => VRM, private stations: () => ComputerStation[]) {}
  snapshot(): ComputerState | null { return this.station?.snapshot() ?? null; }

  prepare(target: number[], sigma: number): Pose {
    const center = new THREE.Vector3().fromArray(target), vrm = this.vrm();
    this.station = matchComputer(this.stations(), center, sigma);
    this.forward.copy(center).sub(point(vrm, 'hips')); this.forward.y = 0; this.forward.normalize();
    const side = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), this.forward);
    this.targets = [1, -1].map(sign => center.clone().addScaledVector(side, sign*.075));
    this.leanToReach(side);
    this.faceDisplay();
    if (!this.station.contact(this.palms())) throw new Error('双手未触到键盘，不能开始游戏');
    return capturePose(vrm);
  }

  private faceDisplay(): void {
    const vrm = this.vrm(), head = contactBone(vrm, 'head'), neck = contactBone(vrm, 'neck');
    const rest = measureVrmRest(vrm).get('head');
    if (!rest) throw new Error('使用电脑需要头部方向标定');
    const up = new THREE.Vector3(0, 1, 0), side = new THREE.Vector3().crossVectors(up, this.forward);
    const desired = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(side, up, this.forward)).multiply(rest);
    const correction = desired.clone().multiply(head.getWorldQuaternion(new THREE.Quaternion()).invert());
    const neckCorrection = new THREE.Quaternion().slerp(correction, .45);
    setWorldRotation(neck, neckCorrection.multiply(neck.getWorldQuaternion(new THREE.Quaternion())));
    setWorldRotation(head, desired);
  }

  private leanToReach(axis: THREE.Vector3): void {
    const vrm = this.vrm(), base = capturePose(vrm), spine = contactBone(vrm, 'spine');
    const rotation = spine.getWorldQuaternion(new THREE.Quaternion());
    for (let angle = 0; angle <= 40; angle += 5) {
      applyPose(base);
      setWorldRotation(spine, new THREE.Quaternion().setFromAxisAngle(axis, THREE.MathUtils.degToRad(angle)).multiply(rotation));
      try { this.hands(0); return; }
      catch (error) {
        if (angle === 40) { applyPose(base); throw error; }
      }
    }
  }

  private palms(): THREE.Vector3[] { return [palmPosition(this.vrm(), 'left'), palmPosition(this.vrm())]; }

  private hands(elapsed: number): void {
    const vrm = this.vrm();
    for (const [index, side] of (['left', 'right'] as const).entries()) {
      const target = this.targets[index].clone(); target.y += .004*Math.sin(elapsed*12+index*Math.PI);
      const pole = this.forward.clone().negate().add(new THREE.Vector3(0, -1, 0));
      if (reachPalm(vrm, target, pole, side, this.palmRotation(side)) > .025) throw new Error('键盘超出坐姿手臂可达范围');
    }
  }

  private palmRotation(side: 'left' | 'right'): THREE.Quaternion {
    const vrm = this.vrm(), hand = contactBone(vrm, `${side}Hand`);
    const along = point(vrm, `${side}MiddleProximal`).sub(point(vrm, `${side}Hand`)).normalize();
    const across = point(vrm, `${side}IndexProximal`).sub(point(vrm, `${side}LittleProximal`));
    across.addScaledVector(along, -across.dot(along)).normalize();
    const source = new THREE.Matrix4().makeBasis(along, across, new THREE.Vector3().crossVectors(along, across));
    const lateral = new THREE.Vector3().crossVectors(this.forward, new THREE.Vector3(0, 1, 0)).multiplyScalar(side === 'left' ? 1 : -1);
    const destination = new THREE.Matrix4().makeBasis(this.forward, lateral, new THREE.Vector3().crossVectors(this.forward, lateral));
    const delta = new THREE.Quaternion().setFromRotationMatrix(destination.multiply(source.invert()));
    return delta.multiply(hand.getWorldQuaternion(new THREE.Quaternion()));
  }

  start(duration: number): Promise<void> {
    if (!this.station) return Promise.reject(new Error('电脑尚未完成接触校验'));
    this.station.start();
    return new Promise((resolve, reject) => { this.session = { elapsed: 0, duration, resolve, reject }; });
  }

  update(dt: number): void {
    const session = this.session, station = this.station;
    if (!session || !station) return;
    try {
      const step = Math.max(0, Math.min(.05, dt)); session.elapsed += step;
      this.hands(session.elapsed);
      if (!station.contact(this.palms())) throw new Error('键盘接触丢失，停止输入');
      station.update(step, true);
      if (session.elapsed >= session.duration) {
        this.session = null; station.stop(true); session.resolve();
      }
    } catch (error) {
      this.session = null; station.stop();
      session.reject(error instanceof Error ? error : new Error(String(error)));
    }
  }

  cancel(): void {
    const session = this.session; this.session = null; this.station?.stop();
    session?.reject(new ComputerPlayCancelled('电脑操作已取消'));
  }

  reset(): void { this.cancel(); this.station?.reset(); this.station = null; this.targets = []; }
}

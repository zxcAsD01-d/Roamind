import * as THREE from 'three';
import type { VRM } from '@pixiv/three-vrm';
import { measureVrmRest } from './retarget';

const UP = new THREE.Vector3(0, 1, 0);
const wrap = (value: number): number => Math.atan2(Math.sin(value), Math.cos(value));
const radians = THREE.MathUtils.degToRad;
export const HEAD_YAW_LIMIT = 75;
export const HEAD_PITCH_LIMIT = 30;
export const headActions = (['yaw', 'pitch'] as const).flatMap(axis => {
  const limit = axis === 'yaw' ? HEAD_YAW_LIMIT : HEAD_PITCH_LIMIT;
  return Array.from({ length: limit*2/5+1 }, (_, index) => {
    const degrees = index*5-limit;
    return { id: 'look_'+axis+'_'+(degrees >= 0 ? '+' : '')+degrees, axis, degrees };
  });
});

/** A reversible upper-body layer. Never changes root motion or leg poses. */
export class HeadLookController {
  private saved: { bone: THREE.Object3D; rotation: THREE.Quaternion }[] = [];
  private yaw = 0;
  private velocity = 0;
  private target: number | null = null;
  private enabled = false;
  private initialize = true;
  private pitch = 0;
  private pitchVelocity = 0;
  private pitchTarget = 0;

  aim(heading: number, bodyHeading = heading, pitchDegrees = 0): void {
    const degrees = THREE.MathUtils.radToDeg(wrap(heading-bodyHeading));
    if (Math.abs(degrees) > HEAD_YAW_LIMIT+.01 || Math.abs(pitchDegrees) > HEAD_PITCH_LIMIT)
      throw new Error('观察目标超出头颈活动范围');
    const snapped = Math.sign(degrees)*Math.floor(Math.abs(degrees)/5+.5)*5;
    this.target = bodyHeading+radians(snapped);
    this.pitchTarget = radians(Math.sign(pitchDegrees)*Math.floor(Math.abs(pitchDegrees)/5+.5)*5);
    this.enabled = true;
  }
  release(): void {
    this.target = null; this.enabled = false; this.initialize = true;
    this.yaw = 0; this.velocity = 0; this.pitch = 0; this.pitchVelocity = 0; this.pitchTarget = 0;
    this.restore();
  }

  restore(): void {
    for (const { bone, rotation } of this.saved) bone.quaternion.copy(rotation);
    this.saved = [];
  }

  update(vrm: VRM, delta: number, bodyHeading: number, walking: boolean, clock: number): void {
    if (!this.enabled) return;
    if (this.initialize) {
      this.yaw = wrap(this.heading(vrm)-bodyHeading); this.velocity = 0; this.initialize = false;
    }
    const limit = radians(walking ? 30 : HEAD_YAW_LIMIT);
    const desired = this.target === null ? radians(walking ? 5 : 0)*Math.sin(clock*.85)
      : wrap(this.target-bodyHeading);
    const target = THREE.MathUtils.clamp(desired, -limit, limit);
    const acceleration = THREE.MathUtils.clamp(64*(target-this.yaw)-16*this.velocity, -radians(220), radians(220));
    this.velocity = THREE.MathUtils.clamp(this.velocity+acceleration*delta, -radians(65), radians(65));
    this.yaw += this.velocity*delta;
    this.apply(vrm, bodyHeading+this.yaw);
    this.pitchVelocity = THREE.MathUtils.clamp(this.pitchVelocity+(64*(this.pitchTarget-this.pitch)-16*this.pitchVelocity)*delta,
      -radians(45), radians(45));
    this.pitch += this.pitchVelocity*delta;
    this.applyPitch(vrm);
  }

  explore(): void { this.target = null; this.pitchTarget = 0; this.enabled = true; }

  settled(bodyHeading: number): boolean {
    return this.target !== null && Math.abs(wrap(this.target-bodyHeading-this.yaw)) < radians(2)
      && Math.abs(this.velocity) < radians(6) && Math.abs(this.pitchTarget-this.pitch) < radians(2)
      && Math.abs(this.pitchVelocity) < radians(6);
  }

  diagnostics(): { yawDegrees: number; speedDegrees: number; target: number | null; enabled: boolean } {
    return { yawDegrees: THREE.MathUtils.radToDeg(this.yaw), speedDegrees: THREE.MathUtils.radToDeg(this.velocity),
      target: this.target, enabled: this.enabled };
  }

  private apply(vrm: VRM, heading: number): void {
    const correction = wrap(heading-this.heading(vrm));
    const head = vrm.humanoid.getNormalizedBoneNode('head')!;
    const neck = vrm.humanoid.getNormalizedBoneNode('neck');
    if (neck) this.rotateWorld(neck, correction*.35);
    this.rotateWorld(head, correction*(neck ? .65 : 1));
  }

  private applyPitch(vrm: VRM): void {
    const head = vrm.humanoid.getNormalizedBoneNode('head')!;
    const neck = vrm.humanoid.getNormalizedBoneNode('neck');
    for (const [bone, weight] of [[neck, .35], [head, neck ? .65 : 1]] as const) {
      if (!bone) continue;
      if (!this.saved.some(saved => saved.bone === bone)) this.saved.push({ bone, rotation: bone.quaternion.clone() });
      bone.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), this.pitch*weight));
      bone.updateWorldMatrix(false, true);
    }
  }

  private heading(vrm: VRM): number {
    const head = vrm.humanoid.getNormalizedBoneNode('head');
    const rest = measureVrmRest(vrm).get('head');
    if (!head || !rest) throw new Error('转头控制需要已标定的头部骨骼');
    head.updateWorldMatrix(true, false);
    const orientation = head.getWorldQuaternion(new THREE.Quaternion()).multiply(rest.clone().invert());
    const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(orientation);
    return Math.atan2(forward.x, forward.z);
  }

  private rotateWorld(bone: THREE.Object3D, angle: number): void {
    this.saved.push({ bone, rotation: bone.quaternion.clone() });
    const parent = bone.parent?.getWorldQuaternion(new THREE.Quaternion()) ?? new THREE.Quaternion();
    const axis = UP.clone().applyQuaternion(parent.invert());
    bone.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(axis, angle));
    bone.updateWorldMatrix(false, true);
  }
}

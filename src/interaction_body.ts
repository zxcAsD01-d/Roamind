import * as THREE from 'three';
import type { VRM } from '@pixiv/three-vrm';
import type { Collider } from './house';
import type { ComputerStation, ComputerState } from './computer';
import { ComputerPlayer, ComputerPlayCancelled } from './computer_player';
import { measureVrmRest } from './retarget';
import { contactBone, bodyCollisionProfile } from './contact_rig';
import { InteractionProps, type HeldProp } from './interaction_props';
import { CONSUMPTION_STYLES, consumptionPose, itemContact, mouthPosition, type ConsumptionState } from './consumption_motion';
import { applyPose, blendPose, capturePose, plantFeet, palmPosition, reachPalm, point, setWorldPosition, setWorldRotation, smoothStep, solveLimb, type Pose } from './interaction_pose';

export type InteractionNode = { kind: 'support' | 'grasp' | 'workstation'; controlPoint: number[]; controlAxis: number[];
  size: number[]; sigma: number; observedAt: number; source: string; frame: string; seat?: InteractionNode };
export type BodyOperation = 'sit' | 'lie' | 'pickup' | 'stand' | 'release' | 'play_computer' | 'eat' | 'drink';
export type InteractionState = { posture: 'stand' | 'sit' | 'lie'; supported: boolean; heldObject: string | null;
  objectId: string | null; lastCommand: string | null; transitioning: boolean; supportError: number | null; graspError: number | null;
  computer?: ComputerState | null; activity?: 'computer' | null; consumption?: ConsumptionState | null };
type Context = { vrm: () => VRM | null; root: THREE.Object3D; obstacles: () => Collider[]; props: () => InteractionProps | null;
  computers?: () => ComputerStation[] };
type Transition = { from: Pose; to: Pose; elapsed: number; duration: number; feet: THREE.Vector3[] | null;
  feetRotations: THREE.Quaternion[]; forward: THREE.Vector3; resolve: () => void; reject: (error: Error) => void };
class InteractionCancelled extends Error {}
const up = new THREE.Vector3(0, 1, 0);

export class InteractionBody {
  private standing: Pose | null = null;
  private stable: Pose | null = null;
  private seated: Pose | null = null;
  private lifted: Pose | null = null;
  private standingFeet: THREE.Vector3[] = [];
  private footRotations: THREE.Quaternion[] = [];
  private restForward = new THREE.Vector3(0, 0, 1);
  private transition: Transition | null = null;
  private held: HeldProp | null = null;
  private holdReturn: Pose | null = null;
  private revision = 0;
  private epoch = 0;
  private operationRevision = 0;
  private recovering = false;
  private task: Promise<void> | null = null;
  private computer: ComputerPlayer;
  private consumptionUses = { eat: 0, drink: 0 };
  private state: InteractionState = { posture: 'stand', supported: false, heldObject: null,
    objectId: null, lastCommand: null, transitioning: false, supportError: null, graspError: null };
  constructor(private context: Context) { this.computer = new ComputerPlayer(() => this.vrm(), () => this.context.computers?.() ?? []); }
  get ownsPose(): boolean { return this.task !== null || this.transition !== null || this.stable !== null; }
  get busy(): boolean { return this.task !== null || this.transition !== null; }
  snapshot(): InteractionState { const computer = this.computer.snapshot();
    return { ...this.state, transitioning: this.busy, computer, activity: computer?.active ? 'computer' : null }; }
  private vrm(): VRM { const vrm = this.context.vrm(); if (!vrm) throw new Error('角色骨架尚未就绪'); return vrm; }

  execute(command: string, operation: BodyOperation, objectId?: string, node?: InteractionNode, duration = 30): Promise<void> {
    if (this.busy) return Promise.reject(new Error('交互姿态仍在过渡'));
    const task = this.perform(command, operation, objectId, node, duration); this.task = task;
    const clear = (): void => { if (this.task === task) this.task = null; };
    void task.then(clear, clear);
    return task;
  }

  private async perform(command: string, operation: BodyOperation, objectId?: string, node?: InteractionNode, duration = 30): Promise<void> {
    const revision = ++this.revision; this.operationRevision = revision;
    if (operation === 'stand') await this.stand();
    else if (operation === 'release') await this.release();
    else {
      this.validateNode(node, operation);
      if (operation === 'play_computer') await this.playComputer(node!, duration);
      else if (operation === 'eat' || operation === 'drink') await this.consume(command, operation, node!, revision, objectId);
      else if (operation === 'pickup') await this.pickup(node!, revision, objectId);
      else await this.rest(operation, node!);
    }
    if (revision !== this.revision) throw new InteractionCancelled('交互已被替换');
    this.state.lastCommand = command;
    if (objectId) this.state.objectId = objectId;
  }

  private validateNode(node: InteractionNode | undefined, operation: BodyOperation): void {
    if (!node || node.frame !== 'control-world-y-up' || node.source !== 'measured-stereo-surface'
      || node.kind !== (operation === 'play_computer' ? 'workstation' : ['pickup', 'eat', 'drink'].includes(operation) ? 'grasp' : 'support')
      || ![node.controlPoint, node.controlAxis, node.size].every(v => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite))
      || node.size.some(v => v < 0) || Math.abs(node.controlAxis[1]) > .05
      || Math.abs(new THREE.Vector3().fromArray(node.controlAxis).length()-1) > .02
      || !Number.isFinite(node.observedAt) || Date.now()/1000-node.observedAt > (['pickup', 'eat', 'drink'].includes(operation) ? 2 : 10) || node.observedAt-Date.now()/1000 > .1
      || !Number.isFinite(node.sigma) || node.sigma < 0 || node.sigma > .03) throw new Error('交互节点缺失、过期或精度不足');
  }

  private async playComputer(node: InteractionNode, duration: number): Promise<void> {
    if (!Number.isFinite(duration) || duration < 5 || duration > 120) throw new Error('电脑操作时长需为 5–120 秒');
    if (this.held) throw new Error('请先放回手中的物品');
    this.validateNode(node.seat, 'sit');
    const seat = node.seat!, target = new THREE.Vector3().fromArray(node.controlPoint);
    const facing = this.forward(new THREE.Vector3().fromArray(seat.controlPoint));
    const toward = target.clone().sub(new THREE.Vector3().fromArray(seat.controlPoint)); toward.y = 0;
    if (facing.dot(toward.normalize()) < .9) throw new Error('需从座椅朝向键盘的一侧接近');
    await this.rest('sit', seat);
    const seated = capturePose(this.vrm()), epoch = this.epoch;
    try {
      const inputPose = this.computer.prepare(node.controlPoint, node.sigma); applyPose(seated);
      await this.move(inputPose, 1.5, null, facing);
      this.checkClearance(); this.stable = capturePose(this.vrm());
      await this.computer.start(duration);
      await this.move(seated, 1.2, null, facing); this.stable = seated;
    } catch (error) {
      this.computer.cancel();
      if (epoch === this.epoch) { await this.recover(seated); this.stable = seated; }
      if (error instanceof ComputerPlayCancelled) throw new InteractionCancelled(error.message);
      throw error;
    }
  }

  private forward(target: THREE.Vector3): THREE.Vector3 {
    const direction = this.context.root.getWorldPosition(new THREE.Vector3()).sub(target); direction.y = 0;
    if (direction.length() < .05) throw new Error('需从支撑面外侧接近');
    return direction.normalize();
  }

  private support(point: THREE.Vector3): number {
    const error = this.context.obstacles().filter(box => box.kind === 'furniture'
      && Math.abs(point.x-box.center[0]) <= box.size[0]/2-.025
      && Math.abs(point.z-box.center[2]) <= box.size[2]/2-.025)
      .reduce((best, box) => Math.min(best, Math.abs(point.y-(box.center[1]+box.size[1]/2))), Infinity);
    if (error > .045) throw new Error('实测支撑点未通过执行支撑校验，拒绝悬空坐卧');
    return error;
  }

  private async rest(posture: 'sit' | 'lie', node: InteractionNode): Promise<void> {
    if (this.state.posture !== 'stand') await this.stand();
    const epoch = this.epoch;
    const vrm = this.vrm(), original = capturePose(vrm);
    const feet = [point(vrm, 'leftFoot'), point(vrm, 'rightFoot')];
    const target = new THREE.Vector3().fromArray(node.controlPoint), forward = this.forward(target);
    this.standingFeet = feet; this.restForward.copy(forward);
    this.footRotations = ['leftFoot', 'rightFoot'].map(name => contactBone(vrm, name as 'leftFoot' | 'rightFoot').getWorldQuaternion(new THREE.Quaternion()));
    const floor = Math.min(...feet.map(foot => foot.y));
    if (target.y-floor < .18 || target.y-floor > bodyCollisionProfile(vrm).height*.55) throw new Error('支撑面高度不适合当前角色坐卧');
    try {
      this.seatedPose(target, forward, node, feet);
      this.seated = capturePose(vrm);
      if (posture === 'lie') {
        this.liftFeet(forward); this.lifted = capturePose(vrm);
        applyPose(original); this.lyingPose(new THREE.Vector3().fromArray(node.controlPoint), node);
      }
      const goal = capturePose(vrm);
      this.state.supportError = this.support(target);
      this.standing = original;
      applyPose(original);
      await this.move(this.seated, 2.2, feet, forward);
      if (posture === 'lie') {
        await this.move(this.lifted!, 1.2, null, forward);
        await this.move(goal, 3.2, null, forward);
      }
      this.checkClearance();
      this.stable = capturePose(vrm); this.state.posture = posture; this.state.supported = true;
    } catch (error) {
      if (epoch === this.epoch) { await this.recover(original); this.stable = null; this.standing = null; this.state.supportError = null; }
      throw error;
    }
  }

  private seatedPose(target: THREE.Vector3, forward: THREE.Vector3, node: InteractionNode, feet: THREE.Vector3[]): void {
    const vrm = this.vrm();
    const inset = Math.max(0, Math.min(node.size[0]/2-.10, node.size[2]/2-.10));
    target.addScaledVector(forward, inset);
    this.support(target);
    const hips = contactBone(vrm, 'hips');
    setWorldRotation(hips, new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(
      new THREE.Vector3().crossVectors(up, forward), up, forward)).multiply(measureVrmRest(vrm).get('hips')!));
    setWorldPosition(hips, target.clone().addScaledVector(up, .09));
    if (plantFeet(vrm, feet, forward, this.footRotations) > .025) throw new Error('坐姿脚部超出腿长范围，需要更近的落脚点');
    this.relaxHands(forward);
  }

  private liftFeet(forward: THREE.Vector3): void {
    const vrm = this.vrm(), hips = point(vrm, 'hips'), side = new THREE.Vector3().crossVectors(up, forward);
    for (const [name, sign] of [['left', 1], ['right', -1]] as const) {
      const target = hips.clone().addScaledVector(forward, .22).addScaledVector(side, sign*.09).addScaledVector(up, .10);
      solveLimb(vrm, [`${name}UpperLeg`, `${name}LowerLeg`, `${name}Foot`], target, forward.clone().add(up));
    }
  }

  private lyingPose(target: THREE.Vector3, node: InteractionNode): void {
    const vrm = this.vrm(), height = bodyCollisionProfile(vrm).height;
    if (Math.max(node.size[0], node.size[2]) < height*.85) throw new Error('支撑面长度不足以躺下');
    const along = new THREE.Vector3().fromArray(node.controlAxis); along.y = 0; along.normalize();
    if (node.size[2] > node.size[0]) along.cross(up).normalize();
    const right = new THREE.Vector3().crossVectors(along, up).normalize();
    const rotation = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, along, up));
    const hips = contactBone(vrm, 'hips');
    setWorldRotation(hips, rotation.multiply(measureVrmRest(vrm).get('hips')!)); setWorldPosition(hips, target.clone().addScaledVector(up, .15));
    for (const side of ['left', 'right'] as const) {
      contactBone(vrm, `${side}UpperLeg`).quaternion.identity(); contactBone(vrm, `${side}LowerLeg`).quaternion.identity();
    }
    hips.updateWorldMatrix(true, true); this.relaxLyingHands(along, right);
    for (const name of ['head', 'leftFoot', 'rightFoot'] as const) {
      const probe = point(vrm, name); probe.y = target.y; this.support(probe);
    }
  }

  private relaxLyingHands(along: THREE.Vector3, lateral: THREE.Vector3): void {
    const vrm = this.vrm(), hips = point(vrm, 'hips');
    for (const [side, sign] of [['left', 1], ['right', -1]] as const) {
      const target = hips.clone().addScaledVector(lateral, sign*.18).addScaledVector(along, .02).addScaledVector(up, -.04);
      const pole = lateral.clone().multiplyScalar(sign*.7).addScaledVector(along, -.5);
      solveLimb(vrm, [`${side}UpperArm`, `${side}LowerArm`, `${side}Hand`], target, pole);
    }
  }

  private relaxHands(forward: THREE.Vector3): void {
    const vrm = this.vrm(), hips = point(vrm, 'hips');
    const orientation = contactBone(vrm, 'hips').getWorldQuaternion(new THREE.Quaternion()).multiply(measureVrmRest(vrm).get('hips')!.clone().invert());
    const lateral = new THREE.Vector3(1, 0, 0).applyQuaternion(orientation);
    const bodyUp = new THREE.Vector3(0, 1, 0).applyQuaternion(orientation);
    for (const [side, sign] of [['left', 1], ['right', -1]] as const) {
      const target = hips.clone().addScaledVector(forward, .22).addScaledVector(lateral, sign*.14).addScaledVector(bodyUp, .02);
      solveLimb(vrm, [`${side}UpperArm`, `${side}LowerArm`, `${side}Hand`], target, lateral.clone().multiplyScalar(sign*.3).addScaledVector(bodyUp, -1));
    }
  }

  private async pickup(node: InteractionNode, revision: number, objectId?: string): Promise<void> {
    if (this.held) throw new Error('右手已经持有物品，请先放回');
    const vrm = this.vrm(), props = this.context.props();
    if (!props) throw new Error('场景没有可移动道具适配器');
    const target = new THREE.Vector3().fromArray(node.controlPoint), item = props.match(target, node.sigma);
    const original = capturePose(vrm), previousStable = this.stable, epoch = this.epoch;
    const feet = [point(vrm, 'leftFoot'), point(vrm, 'rightFoot')], forward = target.clone().sub(point(vrm, 'hips')); forward.y = 0; forward.normalize();
    try {
      this.reachPose(target, feet, forward);
      const reaching = capturePose(vrm); applyPose(original);
      await this.move(reaching, 1.8, this.state.posture === 'stand' ? feet : null, forward);
      if (revision !== this.revision) throw new InteractionCancelled('拿取已取消');
      const reached = capturePose(vrm); // Includes the executed foot orientation constraint.
      this.state.graspError = palmPosition(vrm).distanceTo(target);
      if (this.state.graspError > .045) throw new Error('手与抓取点尚未收敛');
      props.attach(item, contactBone(vrm, 'rightHand'), palmPosition(vrm)); this.held = item;
      this.state.heldObject = item.definition.id; this.state.objectId = objectId ?? null; this.holdReturn = original;
      applyPose(original); this.holdPose(); const holding = capturePose(vrm); applyPose(reached);
      await this.move(holding, 1.5, this.state.posture === 'stand' ? feet : null, forward);
      this.stable = this.state.posture === 'stand' ? null : capturePose(vrm);
    } catch (error) {
      if (epoch === this.epoch) {
        const current = capturePose(vrm); applyPose(original);
        if (this.held === item) this.holdPose();
        const safe = capturePose(vrm); applyPose(current);
        await this.recover(safe);
        this.stable = previousStable ? safe : null;
      }
      throw error;
    }
  }

  private reachPose(target: THREE.Vector3, feet: THREE.Vector3[], forward: THREE.Vector3): void {
    const vrm = this.vrm(), hips = contactBone(vrm, 'hips');
    if (this.state.posture === 'stand') {
      const shoulder = point(vrm, 'rightUpperArm');
      const lower = THREE.MathUtils.clamp(shoulder.y-target.y-.24, 0, .38);
      setWorldPosition(hips, point(vrm, 'hips').addScaledVector(up, -lower).addScaledVector(forward, .08));
      if (plantFeet(vrm, feet, forward) > .035) throw new Error('下蹲抓取无法保持地板支撑');
    }
    const pole = forward.clone().negate().add(new THREE.Vector3(-.3, -.5, 0));
    if (reachPalm(vrm, target, pole) > .02) throw new Error('抓取点超出手臂可达范围');
  }

  private holdPose(): void {
    const vrm = this.vrm(), chest = contactBone(vrm, 'chest');
    const orientation = chest.getWorldQuaternion(new THREE.Quaternion()).multiply(measureVrmRest(vrm).get('chest')!.clone().invert());
    const target = new THREE.Vector3(-.12, -.08, .18).applyQuaternion(orientation).add(point(vrm, 'chest'));
    reachPalm(vrm, target, new THREE.Vector3(-1, -.5, 0));
  }

  private async consume(command: string, operation: 'eat' | 'drink', node: InteractionNode, revision: number, objectId?: string): Promise<void> {
    if (!objectId) throw new Error('吃喝需要视觉物体标识');
    const props = this.context.props();
    if (!props) throw new Error('缺少食物/饮水执行适配器');
    const item = props.match(new THREE.Vector3().fromArray(node.controlPoint), node.sigma);
    if (item.definition.kind !== (operation === 'eat' ? 'food' : 'water')) throw new Error('视觉目标与实际食物/饮水类型不符');
    this.state.consumption = null;
    await this.pickup(node, revision, objectId);
    const holding = capturePose(this.vrm()), epoch = this.epoch;
    const feet = [point(this.vrm(), 'leftFoot'), point(this.vrm(), 'rightFoot')];
    const forward = item.home.clone().sub(point(this.vrm(), 'hips')); forward.y = 0; forward.normalize();
    const variant = this.consumptionUses[operation]++%CONSUMPTION_STYLES.length;
    try {
      const error = await this.consumeSequence(item, operation, variant, feet, forward);
      const current = capturePose(this.vrm()); applyPose(holding); const lowered = capturePose(this.vrm()); applyPose(current);
      await this.move(lowered, 1.2, this.state.posture === 'stand' ? feet : null, forward);
      await this.release(feet, forward);
      if (revision !== this.revision) throw new InteractionCancelled('吃喝已取消');
      const remaining = props.consume(item, operation);
      this.state.consumption = { commandId: command, objectId, operation, completed: true, returned: true,
        mouthError: error, graspError: this.state.graspError ?? Infinity, remaining, variant };
    } catch (error) {
      if (epoch === this.epoch) { await this.recover(holding); this.stable = this.state.posture === 'stand' ? null : holding; }
      throw error;
    }
  }

  private async consumeSequence(item: HeldProp, operation: 'eat' | 'drink', variant: number,
    feet: THREE.Vector3[], forward: THREE.Vector3): Promise<number> {
    const vrm = this.vrm(), style = CONSUMPTION_STYLES[variant];
    let error = 0;
    for (let step = 0; step < 3; step++) {
      const original = capturePose(vrm);
      const result = consumptionPose(vrm, item, operation, step === 1 ? style.tilt : style.tilt*.45);
      applyPose(original);
      await this.move(result.pose, step === 0 ? style.lift : style.contact, this.state.posture === 'stand' ? feet : null, forward);
      item.mesh.updateWorldMatrix(true, true);
      const measured = item.mesh.localToWorld(itemContact(item, operation)).distanceTo(mouthPosition(vrm));
      if (!Number.isFinite(measured) || measured > .035) throw new Error('实际口部接触未收敛');
      error = Math.max(error, measured);
      await this.move(capturePose(vrm), style.pause, this.state.posture === 'stand' ? feet : null, forward);
    }
    return error;
  }

  private async release(feetOverride: THREE.Vector3[] | null = null, forwardOverride: THREE.Vector3 | null = null): Promise<void> {
    const item = this.held, props = this.context.props();
    if (!item || !props) throw new Error('当前没有持有物品');
    const epoch = this.epoch;
    const vrm = this.vrm(), original = capturePose(vrm), hand = contactBone(vrm, 'rightHand');
    const feet = feetOverride ?? [point(vrm, 'leftFoot'), point(vrm, 'rightFoot')];
    const forward = forwardOverride ?? item.home.clone().sub(point(vrm, 'hips')); forward.y = 0; forward.normalize();
    const offset = item.mesh.getWorldPosition(new THREE.Vector3()).sub(palmPosition(vrm));
    try {
      this.reachPose(item.home.clone().sub(offset), feet, forward);
      const releasing = capturePose(vrm); applyPose(original);
      await this.move(releasing, 1.8, this.state.posture === 'stand' ? feet : null, forward);
      if (item.mesh.getWorldPosition(new THREE.Vector3()).distanceTo(item.home) > .065) throw new Error('物品未回到放置位置');
      props.release(item); item.mesh.position.copy(item.home); item.mesh.quaternion.identity();
      this.held = null; this.state.heldObject = null;
      const arm = new Set(['rightUpperArm', 'rightLowerArm', 'rightHand'].map(name => contactBone(vrm, name as 'rightUpperArm' | 'rightLowerArm' | 'rightHand')));
      const destination = original.map(row => arm.has(row.node) ? this.holdReturn?.find(old => old.node === row.node) ?? row : row);
      await this.move(destination, 1.5, this.state.posture === 'stand' ? feet : null, forward);
      this.stable = this.state.posture === 'stand' ? null : capturePose(vrm);
    } catch (error) { if (epoch === this.epoch) await this.recover(original); throw error; }
    hand.updateWorldMatrix(true, true);
  }

  async stand(): Promise<void> {
    if (this.state.posture === 'stand') return;
    if (!this.standing) throw new Error('缺少起身支撑姿态');
    const target = this.standing, original = capturePose(this.vrm()), epoch = this.epoch;
    try {
      if (this.state.posture === 'lie' && this.seated && this.lifted) {
        await this.move(this.lifted, 3.2, null, this.restForward);
        await this.move(this.seated, 1.2, null, this.restForward);
      }
      await this.move(target, 2.4, this.standingFeet, this.restForward);
      this.stable = null; this.standing = null; this.seated = null;
      this.state.posture = 'stand'; this.state.supported = false; this.state.supportError = null;
    } catch (error) { if (epoch === this.epoch) await this.recover(original); throw error; }
  }

  private async recover(pose: Pose): Promise<void> {
    this.recovering = true;
    try { await this.move(pose, .8, null, this.restForward); }
    finally { this.recovering = false; }
  }

  private checkClearance(): void {
    const vrm = this.vrm();
    const names = ['hips', 'spine', 'chest', 'neck', 'head', 'leftLowerLeg', 'rightLowerLeg',
      'leftFoot', 'rightFoot', 'leftLowerArm', 'rightLowerArm', 'leftHand', 'rightHand'] as const;
    for (const name of names) {
      const p = point(vrm, name);
      const hit = this.context.obstacles().find(box => p.x > box.center[0]-box.size[0]/2+.015
        && p.x < box.center[0]+box.size[0]/2-.015 && p.z > box.center[2]-box.size[2]/2+.015
        && p.z < box.center[2]+box.size[2]/2-.015 && p.y > box.center[1]-box.size[1]/2+.015
        && p.y < box.center[1]+box.size[1]/2-.015);
      if (hit) throw new Error(`交互姿态穿入实体：${name} / ${hit.id}`);
    }
  }

  private move(to: Pose, duration: number, feet: THREE.Vector3[] | null, forward: THREE.Vector3): Promise<void> {
    if (!this.recovering && this.operationRevision !== this.revision) return Promise.reject(new InteractionCancelled('交互已中断'));
    return new Promise((resolve, reject) => { this.transition = { from: capturePose(this.vrm()), to,
      elapsed: 0, duration, feet, forward, feetRotations: ['leftFoot', 'rightFoot'].map(name =>
        contactBone(this.vrm(), name as 'leftFoot' | 'rightFoot').getWorldQuaternion(new THREE.Quaternion())), resolve, reject }; });
  }

  update(dt: number): void {
    const transition = this.transition;
    if (!transition) { if (this.stable && !this.task) applyPose(this.stable);
      if (this.held && !this.task) this.holdPose(); this.computer.update(dt); return; }
    transition.elapsed += Math.min(dt, .05);
    blendPose(transition.from, transition.to, smoothStep(transition.elapsed/transition.duration));
    if (transition.feet) plantFeet(this.vrm(), transition.feet, transition.forward, transition.feetRotations);
    if (transition.elapsed >= transition.duration) {
      this.transition = null;
      const vrm = this.vrm(); vrm.humanoid.update(); vrm.update(0);
      transition.resolve();
    }
  }

  async cancel(): Promise<void> {
    const task = this.task;
    if (!task) return;
    this.revision++;
    this.computer.cancel();
    const transition = this.transition;
    if (transition && !this.recovering) { this.transition = null; transition.reject(new InteractionCancelled('交互动作被取消')); }
    try { await task; }
    catch (error) { if (!(error instanceof InteractionCancelled)) console.error('交互中断前执行失败', error); }
  }
  reset(): void {
    this.epoch++;
    this.computer.reset();
    void this.cancel();
    const transition = this.transition; this.transition = null;
    if (transition) transition.reject(new InteractionCancelled('交互环境已重置'));
    if (this.standing) applyPose(this.standing);
    this.context.props()?.reset(); this.stable = null; this.standing = null; this.held = null; this.holdReturn = null; this.seated = null;
    this.state = { posture: 'stand', supported: false, heldObject: null, objectId: null,
      lastCommand: null, transitioning: false, supportError: null, graspError: null };
  }
}

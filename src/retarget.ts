import * as THREE from "three";
import type { VRM, VRMHumanBoneName } from "@pixiv/three-vrm";

/**
 * SMPL forward kinematics + retargeting onto a VRM humanoid.
 *
 * Conventions (verified against `smplx.lbs.batch_rigid_transform`):
 *   W_child = W_parent * R_local
 * and at SMPL's zero pose every W is the identity, so `W_j(t)` *is* the
 * world-space delta from rest. That makes the retarget simply
 *
 *   targetWorld[bone] = W_source[t] * restWorldVRM[bone]
 *   localWorld[bone]  = inv(targetWorld[parent]) * targetWorld[bone]
 *
 * which needs no knowledge of the avatar's bone axis conventions -- only its
 * rest pose, measured once when the model loads.
 */

export interface SmplMotionJson {
  version: number;
  name: string;
  source: string;
  dataset: string;
  fps: number;
  frames: number;
  duration: number;
  bones: string[];
  parents: number[];
  localRot: number[];
  rootDelta: number[];
  /** Original pelvis position; keeps floor height stable across continuations. */
  rootOrigin?: number[];
  /** Trained source skeleton geometry and contact channels (left heel/toe, right heel/toe). */
  restJoints?: number[];
  footContacts?: number[];
  /**
   * Up axis of the source world.
   *
   * AMASS is Z-up and carries `"zup"` (or omits the field); ARDY's
   * CoreSkeleton27 output is already Y-up with +Z forward and carries
   * `"yup"`, so `SmplClip` must not apply the change of frame to it.
   */
  up?: "zup" | "yup";
  prompt?: string;
}

export interface MotionManifestEntry {
  id: string;
  file: string;
  name: string;
  dataset: string;
  fps: number;
  frames: number;
  duration: number;
  bytes: number;
  /** Folder the clip JSON lives under (defaults to the active list's base). */
  base?: string;
  prompt?: string;
  /** Human-readable label (presets). */
  label?: string;
}

/**
 * AMASS stores its world with **+Z up** (verified across all 13 sub-datasets in
 * the selection: the body's SMPL +Y axis maps to +Z, and the pelvis height -- a
 * constant ~0.95 m for a standing adult -- sits on the Z axis with the ground at
 * z = 0). three.js is +Y up, so the whole motion is re-expressed by rotating
 * -90 degrees about X.
 *
 * Because joints 1..21 are parent-relative, only the *root* needs the change of
 * frame: pre-multiplying the root's local rotation by A turns every world
 * rotation into `A * W`, which is exactly the new-world image of the old one.
 * The SMPL zero pose (`W = I`) therefore lands on a character whose up axis is
 * +Y, i.e. standing -- which is what an upright AMASS frame (`W ~= Rx(+90deg)`)
 * must map to.
 */
const ZUP_TO_YUP = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);

/** Hamilton product `a * b`, matching `THREE.Quaternion.multiply`. */
function mulQuat(
  ax: number, ay: number, az: number, aw: number,
  bx: number, by: number, bz: number, bw: number,
  out: Float32Array, o: number,
): void {
  out[o] = aw * bx + ax * bw + ay * bz - az * by;
  out[o + 1] = aw * by - ax * bz + ay * bw + az * bx;
  out[o + 2] = aw * bz + ax * by - ay * bx + az * bw;
  out[o + 3] = aw * bw - ax * bx - ay * by - az * bz;
}

function axisAngleToQuat(x: number, y: number, z: number, out: Float32Array, o: number): void {
  const angle = Math.hypot(x, y, z);
  if (angle < 1e-8) {
    out[o] = 0;
    out[o + 1] = 0;
    out[o + 2] = 0;
    out[o + 3] = 1;
    return;
  }
  const s = Math.sin(angle / 2) / angle;
  out[o] = x * s;
  out[o + 1] = y * s;
  out[o + 2] = z * s;
  out[o + 3] = Math.cos(angle / 2);
}

/** A decoded clip: per-frame local + world quaternions, and root motion. */
export class SmplClip {
  readonly motion: SmplMotionJson;
  frames: number;
  readonly fps: number;
  /** Joint count — 22 for SMPL, 27 for ARDY's CoreSkeleton27. */
  readonly joints: number;
  /** Whether the source world is Z-up (AMASS) or already Y-up (ARDY). */
  readonly zUp: boolean;
  /** [frames, joints, 4] local quaternions (xyzw). */
  local: Float32Array;
  /** [frames, joints, 4] world quaternions (xyzw), FK'd once up front. */
  world: Float32Array;
  /** [frames, 3] root translation relative to frame 0, metres. */
  root: Float32Array;
  readonly parents: number[];

  private constructor(motion: SmplMotionJson) {
    this.motion = motion;
    this.frames = motion.frames;
    this.fps = motion.fps;
    this.parents = motion.parents;
    this.joints = motion.bones.length;
    this.zUp = (motion.up ?? "zup") === "zup";
    this.local = new Float32Array(this.frames * this.joints * 4);
    this.world = new Float32Array(this.frames * this.joints * 4);
    this.root = new Float32Array(this.frames * 3);

    this.decodeFrames(0);
  }

  private decodeFrames(start: number): void {
    const motion = this.motion;
    const aa = motion.localRot;
    const rootQ = new THREE.Quaternion();
    const swapped = new THREE.Quaternion();
    for (let f = start; f < this.frames; f++) {
      for (let j = 0; j < this.joints; j++) {
        const src = (f * this.joints + j) * 3;
        axisAngleToQuat(aa[src], aa[src + 1], aa[src + 2], this.local, (f * this.joints + j) * 4);
      }

      // Change the root's local rotation into the Y-up frame; the remaining
      // joints are parent-relative and need no adjustment (see ZUP_TO_YUP).
      // ARDY is already Y-up, so it skips this entirely.
      const ro = f * this.joints * 4;
      if (this.zUp) {
        rootQ.set(this.local[ro], this.local[ro + 1], this.local[ro + 2], this.local[ro + 3]);
        swapped.copy(ZUP_TO_YUP).multiply(rootQ);
        this.local[ro] = swapped.x;
        this.local[ro + 1] = swapped.y;
        this.local[ro + 2] = swapped.z;
        this.local[ro + 3] = swapped.w;
      }

      // Parents always have a lower index in these trees, so one pass works.
      for (let j = 0; j < this.joints; j++) {
        const o = (f * this.joints + j) * 4;
        const p = this.parents[j];
        if (p < 0) {
          this.world[o] = this.local[o];
          this.world[o + 1] = this.local[o + 1];
          this.world[o + 2] = this.local[o + 2];
          this.world[o + 3] = this.local[o + 3];
        } else {
          const po = (f * this.joints + p) * 4;
          mulQuat(
            this.world[po], this.world[po + 1], this.world[po + 2], this.world[po + 3],
            this.local[o], this.local[o + 1], this.local[o + 2], this.local[o + 3],
            this.world, o,
          );
        }
      }

      // Root translation: Z-up -> Y-up is (x, y, z) -> (x, z, -y); Y-up sources
      // are already in the target frame.
      const rd = motion.rootDelta;
      if (this.zUp) {
        this.root[f * 3] = rd[f * 3];
        this.root[f * 3 + 1] = rd[f * 3 + 2];
        this.root[f * 3 + 2] = -rd[f * 3 + 1];
      } else {
        this.root[f * 3] = rd[f * 3];
        this.root[f * 3 + 1] = rd[f * 3 + 1];
        this.root[f * 3 + 2] = rd[f * 3 + 2];
      }
    }
  }

  static from(json: SmplMotionJson): SmplClip {
    const joints = json.bones.length;
    if (joints < 1 || json.parents.length !== joints) {
      throw new Error(`bones (${json.bones.length}) and parents (${json.parents.length}) must agree`);
    }
    if (json.localRot.length !== json.frames * joints * 3) {
      throw new Error(`localRot length ${json.localRot.length} != ${json.frames * joints * 3}`);
    }
    if (json.rootDelta.length !== json.frames * 3) {
      throw new Error(`rootDelta length ${json.rootDelta.length} != ${json.frames * 3}`);
    }
    if (json.restJoints && (json.restJoints.length !== joints*3 || !json.restJoints.every(Number.isFinite))) {
      throw new Error('源骨架静止位置数据无效');
    }
    if (json.footContacts && (json.footContacts.length !== json.frames*4
      || !json.footContacts.every(value => value === 0 || value === 1))) {
      throw new Error('足底接触通道数据无效');
    }
    return new SmplClip(json);
  }

  /** Decode only appended frames; grow arrays geometrically to avoid quadratic work. */
  appendFrames(): void {
    const previous = this.frames;
    const next = this.motion.frames;
    if (next < previous || this.motion.localRot.length !== next*this.joints*3
      || this.motion.rootDelta.length !== next*3) throw new Error('追加动作数据长度不正确');
    if (next*this.joints*4 > this.local.length) {
      const capacity = Math.max(next, previous*2);
      const local = new Float32Array(capacity*this.joints*4); local.set(this.local); this.local = local;
      const world = new Float32Array(capacity*this.joints*4); world.set(this.world); this.world = world;
      const root = new Float32Array(capacity*3); root.set(this.root); this.root = root;
    }
    this.frames = next;
    this.decodeFrames(previous);
  }

  rootAt(frame: number, out: THREE.Vector3): THREE.Vector3 {
    const position = Math.max(0, Math.min(this.frames - 1, frame));
    const first = Math.floor(position);
    const next = Math.min(this.frames - 1, first + 1);
    const weight = position - first;
    const offset = first * 3;
    const following = next * 3;
    return out.set(
      THREE.MathUtils.lerp(this.root[offset], this.root[following], weight),
      THREE.MathUtils.lerp(this.root[offset + 1], this.root[following + 1], weight),
      THREE.MathUtils.lerp(this.root[offset + 2], this.root[following + 2], weight),
    );
  }
}

interface BoneBinding {
  bone: string;
  joint: number;
  node: THREE.Object3D;
  depth: number;
}

/**
 * The VRM's *rest* bone orientations, measured once per avatar before any
 * retargeting poses it.
 *
 * This cache is the fix for the "residual SMPL values stacked onto the next
 * clip" bug: `VrmRetargeter` used to read `node.getWorldQuaternion()` in its
 * constructor, which returns the *current* (previous clip's) pose.  The second
 * clip's world delta was then multiplied onto the old pose, twisting the avatar.
 * Measuring once at load -- and never again -- keeps the reference pose clean.
 */
const vrmRestCache = new WeakMap<VRM, Map<string, THREE.Quaternion>>();

export function measureVrmRest(vrm: VRM): Map<string, THREE.Quaternion> {
  const cached = vrmRestCache.get(vrm);
  if (cached) return cached;
  vrm.scene.updateMatrixWorld(true);
  const rest = new Map<string, THREE.Quaternion>();
  const q = new THREE.Quaternion();
  for (const name of Object.keys(vrm.humanoid.humanBones)) {
    const node = vrm.humanoid.getNormalizedBoneNode(name as VRMHumanBoneName);
    if (!node) continue;
    node.getWorldQuaternion(q);
    rest.set(name, q.clone());
  }
  vrmRestCache.set(vrm, rest);
  return rest;
}

/** Binds a clip's SMPL joints to a specific VRM and drives it each frame. */
export class VrmRetargeter {
  private bindings: BoneBinding[] = [];
  private restWorld = new Map<string, THREE.Quaternion>();
  private target = new Map<string, THREE.Quaternion>();
  private parentQuat = new THREE.Quaternion();
  private tmp = new THREE.Quaternion();
  private nextRotation = new THREE.Quaternion();
  /** SMPL joint -> deepest VRM bone it was folded into (for diagnostics). */
  readonly jointToBone = new Map<number, string>();

  constructor(
    private readonly vrm: VRM,
    private readonly clip: SmplClip,
  ) {
    this.rebuild();
  }

  /** Re-derive bindings; call after swapping either the VRM or the clip. */
  rebuild(): void {
    const humanoid = this.vrm.humanoid;
    const motion = this.clip.motion;
    this.bindings = [];
    this.restWorld.clear();
    this.target.clear();
    this.jointToBone.clear();

    // Depth of each SMPL joint, used to decide which joint "owns" a bone when
    // a VRM lacks one (e.g. VRM0 has no upperChest).
    const smplDepth: number[] = new Array(motion.parents.length).fill(0);
    for (let i = 0; i < motion.parents.length; i++) {
      let d = 0;
      let p = motion.parents[i];
      while (p >= 0) {
        d++;
        p = motion.parents[p];
      }
      smplDepth[i] = d;
    }

    // Resolve every SMPL joint to a bone that actually exists on this avatar,
    // walking up the SMPL tree when it does not.
    const best = new Map<string, { joint: number; depth: number }>();
    for (let j = 0; j < motion.bones.length; j++) {
      let joint = j;
      let bone = motion.bones[joint];
      while (bone && !humanoid.getNormalizedBoneNode(bone as VRMHumanBoneName)) {
        const p = motion.parents[joint];
        if (p < 0) {
          bone = "";
          break;
        }
        joint = p;
        bone = motion.bones[joint];
      }
      if (!bone) continue;
      const prev = best.get(bone);
      if (!prev || smplDepth[j] > prev.depth) best.set(bone, { joint: j, depth: smplDepth[j] });
    }

    const rest = measureVrmRest(this.vrm);
    for (const [bone, { joint }] of best) {
      const node = humanoid.getNormalizedBoneNode(bone as VRMHumanBoneName);
      if (!node) continue;
      const restQuat = rest.get(bone);
      if (!restQuat) continue;
      this.restWorld.set(bone, restQuat.clone());
      this.target.set(bone, new THREE.Quaternion());
      this.jointToBone.set(joint, bone);
      this.bindings.push({ bone, joint, node, depth: this.sceneDepth(node) });
    }

    // Ancestors must be written before descendants.
    this.bindings.sort((a, b) => a.depth - b.depth);
  }

  private sceneDepth(node: THREE.Object3D): number {
    let d = 0;
    let p: THREE.Object3D | null = node.parent;
    while (p) {
      d++;
      p = p.parent;
    }
    return d;
  }

  /** Write the clip's frame `frame` onto the VRM. */
  apply(frame: number): void {
    const position = Math.max(0, Math.min(this.clip.frames - 1, frame));
    const first = Math.floor(position);
    const base = first * this.clip.joints * 4;
    const next = Math.min(this.clip.frames - 1, first + 1) * this.clip.joints * 4;
    const weight = position - first;
    const world = this.clip.world;

    for (const b of this.bindings) {
      const o = base + b.joint * 4;
      const rest = this.restWorld.get(b.bone)!;
      // target = W_source * restWorld   (world-space delta applied to rest)
      const following = next + b.joint * 4;
      this.nextRotation.fromArray(world, following);
      this.target.get(b.bone)!.set(world[o], world[o + 1], world[o + 2], world[o + 3])
        .slerp(this.nextRotation, weight).multiply(rest);
    }

    for (const b of this.bindings) {
      const target = this.target.get(b.bone)!;
      const parent = b.node.parent;
      if (parent) {
        parent.updateWorldMatrix(true, false);
        parent.getWorldQuaternion(this.parentQuat);
        // local = inv(parentWorld) * targetWorld
        this.tmp.copy(this.parentQuat).invert().multiply(target);
        b.node.quaternion.copy(this.tmp);
      } else {
        b.node.quaternion.copy(target);
      }
      b.node.updateWorldMatrix(false, true);
    }
  }

  /**
   * The clip's root translation for `frame` (relative to frame 0, Y-up).
   *
   * Returned rather than written onto a node: the caller owns the avatar group
   * so that position and the yaw correction live on the same object, and so the
   * camera can follow the very node that moves.
   */
  rootAt(frame: number): THREE.Vector3 {
    return this.clip.rootAt(frame, this._root ?? (this._root = new THREE.Vector3()));
  }

  private _root?: THREE.Vector3;
}

import * as THREE from 'three';
import { ComputerStation, type ComputerDefinition } from './computer';
import { InteractionProps, type PropDefinition } from './interaction_props';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { wallMaterial, worldSurfaceMaterial } from './wall_texture';
import { segmentBoxDistanceSquared } from './geometry_clearance';

export type Room = { id: string; name: string; kind: string; bounds: number[]; target: number[]; material: string };
export type Collider = { id: string; center: number[]; size: number[]; material: string; kind: string };
export type HouseWorld = {
  id: string; name: string; bounds: number[]; spawn: number[]; rooms: Room[]; objects: Collider[]; props?: PropDefinition[];
  computers?: ComputerDefinition[];
  agent: { radius: number; height: number; skin: number };
};

function texture(kind: string): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 256;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('无法创建场景贴图');
  const wood = kind === 'wood' || kind === 'oak';
  ctx.fillStyle = wood ? '#a98c64' : kind === 'tile' ? '#b1bbb6' : '#d8d3c7';
  ctx.fillRect(0, 0, 256, 256);
  if (kind === 'fabric' || kind === 'linen') {
    ctx.fillStyle = '#e1ded6'; ctx.fillRect(0, 0, 256, 256);
    for (let line = 0; line < 256; line += 4) {
      ctx.fillStyle = '#aaa69d'; ctx.fillRect(line, 0, 1, 256);
      ctx.fillStyle = '#ffffff'; ctx.fillRect(0, line, 256, 1);
    }
  } else if (wood) {
    for (let y = 0; y < 256; y += 32) {
      ctx.fillStyle = y % 64 ? '#ae936d' : '#a18662';
      ctx.fillRect(0, y, 256, 31);
      ctx.strokeStyle = '#655a4944';
      for (let line = 2; line < 30; line += 4) {
        ctx.beginPath(); ctx.moveTo(0, y+line); ctx.bezierCurveTo(80,y+line+2,160,y+line-2,256,y+line); ctx.stroke();
      }
      ctx.fillStyle = '#61594a';
      ctx.fillRect(0, y, 256, 1); ctx.fillRect(y % 64 ? 60 : 190, y, 1, 32);
    }
  } else {
    ctx.strokeStyle = '#626f6644';
    for (let i = 0; i <= 256; i += 64) { ctx.strokeRect(i, 0, 64, 256); ctx.strokeRect(0, i, 256, 64); }
  }
  const map = new THREE.CanvasTexture(canvas);
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = map.wrapT = THREE.RepeatWrapping;
  map.anisotropy = 4;
  return map;
}

function materials(): Map<string, THREE.MeshStandardMaterial> {
  const colors: Record<string, number> = { wall: 0xe0ded2, wood: 0xffffff, oak: 0xc7b08e, tile: 0xffffff,
    screen: 0x142c35, fabric: 0x506f73, linen: 0xece4d5, stone: 0xb2b6b1, metal: 0x707d7e, ceramic: 0xf0eee7, glass: 0xa8ccd0 };
  return new Map(Object.entries(colors).map(([key, color]) => [key, key === 'wall' ? wallMaterial() : worldSurfaceMaterial(new THREE.MeshStandardMaterial({
    color, roughness: key === 'metal' ? .3 : key === 'ceramic' ? .24 : .8, metalness: key === 'metal' ? .65 : 0,
    transparent: key === 'glass', opacity: key === 'glass' ? .2 : 1, depthWrite: key !== 'glass',
    map: ['wood','oak','tile','fabric','linen'].includes(key) ? texture(key) : null,
  }), ['fabric','linen'].includes(key) ? .18 : 2)]));
}

function setSurface(mesh: THREE.Object3D): void {
  mesh.castShadow = true;
  mesh.receiveShadow = true;
}

function detailMaterial(materials: Map<string, THREE.MeshStandardMaterial>, key: string): THREE.MeshStandardMaterial {
  const source = materials.get(key) ?? materials.get('oak');
  if (!source) throw new Error(`缺少场景材质：${key}`);
  return source;
}

export class HouseScene {
  readonly computers: ComputerStation[];
  readonly props: InteractionProps;
  readonly group = new THREE.Group();
  readonly routeGroup = new THREE.Group();
  readonly colliders = new THREE.Group();
  private walls: THREE.InstancedMesh[] = [];
  private fullMatrices: THREE.Matrix4[][] = [];
  private boxGeometry = new THREE.BoxGeometry(1, 1, 1);
  private furnitureGeometry = new RoundedBoxGeometry(1, 1, 1, 2, .035);
  private maps = materials();
  private path: THREE.Line | null = null;
  private body = { radius: .32, height: 1.7 };
  private cutaway: boolean | null = null;
  private fixtures = new THREE.Group();

  constructor(readonly world: HouseWorld, scene: THREE.Scene) {
    this.computers = (world.computers ?? []).map(definition => new ComputerStation(definition));
    this.group.add(...this.computers.map(computer => computer.group));
    this.props = new InteractionProps(world.props ?? []);
    this.group.add(this.props.group);
    this.body = { radius: world.agent.radius, height: world.agent.height };
    const matrix = new THREE.Matrix4();
    const floors = world.rooms.filter(room => room.kind !== 'bathroom' || room.id === 'bathroom3');
    for (const room of floors.concat(world.rooms.filter(room => ['bathroom1','bathroom2'].includes(room.id)))) {
      const [x0,z0,x1,z1] = room.bounds;
      const mesh = new THREE.Mesh(this.boxGeometry, this.maps.get(room.material));
      mesh.scale.set(x1-x0, .08, z1-z0); mesh.position.set((x0+x1)/2, room.kind === 'bathroom' ? -.025 : -.04, (z0+z1)/2);
      setSurface(mesh);
      this.group.add(mesh);
      this.addFloorAccent(room);
    }
    for (const material of this.maps.keys()) {
      for (const kind of ['wall', 'furniture']) {
        const objects = world.objects.filter(obj => obj.material === material && obj.kind === kind);
        if (!objects.length) continue;
        const mesh = new THREE.InstancedMesh(kind === 'wall' ? this.boxGeometry : this.furnitureGeometry, this.maps.get(material), objects.length);
        const originals: THREE.Matrix4[] = [];
        objects.forEach((obj, index) => {
          matrix.compose(new THREE.Vector3().fromArray(obj.center), new THREE.Quaternion(), new THREE.Vector3().fromArray(obj.size));
          originals.push(matrix.clone()); mesh.setMatrixAt(index, matrix);
        });
        if (kind === 'wall') { this.walls.push(mesh); this.fullMatrices.push(originals); }
        mesh.castShadow = material !== 'glass';
        mesh.receiveShadow = true;
        this.group.add(mesh);
      }
    }
    this.addArchitectureDetails();
    this.buildColliderPreview();
    this.colliders.visible = false;
    this.group.add(this.routeGroup, this.colliders, this.fixtures);
    scene.add(this.group);
    this.setCutaway(true);
  }

  private addFloorAccent(room: Room): void {
    if (room.kind !== 'living') return;
    const [x0, z0, x1, z1] = room.bounds;
    const width = Math.min(2.8, Math.max(1.6, (x1-x0)*.42));
    const depth = Math.min(1.8, Math.max(1.1, (z1-z0)*.18));
    const rug = new THREE.Mesh(new THREE.BoxGeometry(width, .012, depth), detailMaterial(this.maps, 'fabric'));
    rug.position.set((x0+x1)/2, .025, z0+(z1-z0)*.58);
    rug.material = (rug.material as THREE.MeshStandardMaterial).clone();
    (rug.material as THREE.MeshStandardMaterial).roughness = .96;
    setSurface(rug);
    this.group.add(rug);
  }

  private addArchitectureDetails(): void {
    const trimMaterial = detailMaterial(this.maps, 'oak');
    const metalMaterial = detailMaterial(this.maps, 'metal');
    const addBox = (size: THREE.Vector3, position: THREE.Vector3, material: THREE.Material, group = this.group): void => {
      const mesh = new THREE.Mesh(this.boxGeometry, material);
      mesh.scale.copy(size); mesh.position.copy(position); setSurface(mesh); group.add(mesh);
    };
    const baseWalls = this.world.objects.filter(object => object.kind === 'wall' && object.material === 'wall'
      && object.center[1]-object.size[1]/2 < .01);
    const trim = new THREE.InstancedMesh(this.boxGeometry, trimMaterial, baseWalls.length);
    baseWalls.forEach((wall, index) => {
      const matrix = new THREE.Matrix4().compose(new THREE.Vector3(wall.center[0], .045, wall.center[2]),
        new THREE.Quaternion(), new THREE.Vector3(wall.size[0]+.012, .09, wall.size[2]+.012));
      trim.setMatrixAt(index, matrix);
    });
    setSurface(trim); this.group.add(trim);
    for (const position of [[-7, 2.5, 7.85], [0, 2.5, 7.85], [7, 2.5, 7.85], [-9.85, 2.5, 3.2], [9.85, 2.5, 3.2]]) {
      addBox(new THREE.Vector3(.12, .18, .8), new THREE.Vector3(position[0], position[1], position[2]), metalMaterial, this.fixtures);
      const lamp = new THREE.Mesh(new THREE.SphereGeometry(.18, 16, 8), new THREE.MeshStandardMaterial({ color: 0xffe8b0, emissive: 0xffb347, emissiveIntensity: 1.8, roughness: .3 }));
      lamp.position.set(position[0], position[1]-.16, position[2]); this.fixtures.add(lamp);
      const light = new THREE.PointLight(0xffd9a0, 3, 5, 2); light.position.copy(lamp.position); this.group.add(light);
    }
  }

  private buildColliderPreview(): void {
    const positions: number[] = [];
    const edges = new THREE.EdgesGeometry(this.boxGeometry);
    const source = edges.getAttribute('position');
    for (const obj of this.world.objects) {
      for (let i=0; i<source.count; i++) positions.push(source.getX(i)*obj.size[0]+obj.center[0],
        source.getY(i)*obj.size[1]+obj.center[1], source.getZ(i)*obj.size[2]+obj.center[2]);
    }
    edges.dispose();
    const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    this.colliders.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0xe9b275 })));
  }

  setCutaway(enabled: boolean): void {
    if (this.cutaway === enabled) return;
    this.cutaway = enabled;
    this.fixtures.visible = !enabled;
    this.walls.forEach((mesh, group) => {
      this.fullMatrices[group].forEach((matrix, index) => {
        const copy = matrix.clone();
        if (enabled) { copy.elements[5] *= .22; copy.elements[13] *= .22; }
        mesh.setMatrixAt(index, copy);
      });
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
    });
  }

  showPath(points: number[][]): void {
    if (this.path) { this.routeGroup.remove(this.path); this.path.geometry.dispose(); (this.path.material as THREE.Material).dispose(); }
    const geometry = new THREE.BufferGeometry().setFromPoints(points.map(point => new THREE.Vector3(point[0], .1, point[2])));
    this.path = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0xb2f5ce }));
    this.routeGroup.add(this.path);
  }

  setBodyProfile(profile: { radius: number; height: number }): void {
    if (!Number.isFinite(profile.radius) || !Number.isFinite(profile.height) || profile.radius <= 0 || profile.height <= 0) {
      throw new Error('角色碰撞标定无效');
    }
    this.body = { ...profile };
  }

  collisionAt(position: THREE.Vector3): string | null {
    const radius = this.body.radius;
    const [x0,z0,x1,z1] = this.world.bounds;
    if (position.x < x0+radius || position.x > x1-radius || position.z < z0+radius || position.z > z1-radius) return '房屋边界';
    return this.world.objects.find(obj => {
      if (obj.center[1]-obj.size[1]/2 >= this.body.height || obj.center[1]+obj.size[1]/2 <= .03) return false;
      const dx = Math.max(0, Math.abs(position.x-obj.center[0])-obj.size[0]/2);
      const dz = Math.max(0, Math.abs(position.z-obj.center[2])-obj.size[2]/2);
      return dx*dx+dz*dz < radius*radius-1e-10;
    })?.id ?? null;
  }

  sweep(from: THREE.Vector3, to: THREE.Vector3): string | null {
    const end = this.collisionAt(to);
    if (end) return end;
    const radius = this.body.radius;
    for (const obj of this.world.objects) {
      if (obj.center[1]-obj.size[1]/2 >= this.body.height || obj.center[1]+obj.size[1]/2 <= .03) continue;
      if (segmentBoxDistanceSquared([from.x, from.z], [to.x, to.z],
        [obj.center[0], obj.center[2]], [obj.size[0]/2, obj.size[2]/2]) < radius*radius-1e-10) return obj.id;
    }
    return null;
  }

  dispose(): void {
    for (const computer of this.computers) computer.dispose();
    this.props.dispose();
    const geometries = new Set<THREE.BufferGeometry>(), materials = new Set<THREE.Material>();
    this.group.traverse(node => {
      if (node instanceof THREE.Mesh || node instanceof THREE.Line) {
        if (node instanceof THREE.InstancedMesh) node.dispose();
        geometries.add(node.geometry);
        for (const material of Array.isArray(node.material) ? node.material : [node.material]) materials.add(material);
      }
    });
    const textures = new Set<THREE.Texture>();
    for (const material of materials) {
      if (material instanceof THREE.MeshStandardMaterial && material.map) textures.add(material.map);
      material.dispose();
    }
    for (const texture of textures) texture.dispose();
    for (const geometry of geometries) geometry.dispose();
    this.group.removeFromParent(); this.group.clear();
  }

  frame(camera: THREE.PerspectiveCamera, controls: { target: THREE.Vector3; maxDistance: number }): void {
    const [x0, z0, x1, z1] = this.world.bounds;
    controls.target.set((x0+x1)/2, .6, (z0+z1)/2);
    const direction = new THREE.Vector3(19, 23, 24).normalize();
    camera.position.copy(controls.target).add(direction); camera.lookAt(controls.target);
    const inverse = camera.quaternion.clone().invert(), vertical = Math.tan(THREE.MathUtils.degToRad(camera.fov/2));
    let distance = 1;
    for (const x of [x0, x1]) for (const z of [z0, z1]) for (const y of [0, 2.8]) {
      const corner = new THREE.Vector3(x, y, z).sub(controls.target).applyQuaternion(inverse);
      distance = Math.max(distance, corner.z+Math.abs(corner.x)/(vertical*camera.aspect), corner.z+Math.abs(corner.y)/vertical);
    }
    camera.position.copy(controls.target).addScaledVector(direction, distance*1.07);
    camera.far = Math.max(150, distance*3); camera.updateProjectionMatrix();
    controls.maxDistance = Math.max(60, distance*1.5);
  }
}

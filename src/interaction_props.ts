import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { propMesh } from './consumable_mesh';
export type PropDefinition = { id: string; label: string; center: number[]; size: number[]; kind?: 'food' | 'water'; portions?: number };
type Prop = { definition: PropDefinition; mesh: THREE.Group; home: THREE.Vector3; held: boolean; remaining: number };

export class InteractionProps {
  readonly group = new THREE.Group();
  private items: Prop[] = [];
  constructor(definitions: PropDefinition[]) {
    for (const definition of definitions) {
      const mesh = propMesh(definition) ?? new THREE.Group();
      if (!definition.kind) {
      const body = new THREE.Mesh(new RoundedBoxGeometry(...definition.size as [number, number, number], 2, .007),
        new THREE.MeshStandardMaterial({ color: 0x252c32, roughness: .4 }));
      mesh.add(body); body.castShadow = true;
      for (let row = 0; row < 6; row++) for (let col = 0; col < 3; col++) {
        const button = new THREE.Mesh(new THREE.SphereGeometry(.005, 8, 4),
          new THREE.MeshStandardMaterial({ color: row === 0 && col === 0 ? 0xcc4638 : 0xabb2ba }));
        button.scale.y = .4; button.position.set((col-1)*.015, definition.size[1]/2+.002, (row-2.5)*.023);
        mesh.add(button);
      }
      }
      mesh.position.fromArray(definition.center); this.group.add(mesh);
      this.items.push({ definition, mesh, home: mesh.position.clone(), held: false, remaining: 1 });
    }
  }
  /** Geometry is checked only against an already selected visual point. It supplies no planning coordinates. */
  match(target: THREE.Vector3, sigma: number): Prop {
    const candidates = this.items.filter(item => !item.held && item.remaining > 0 && item.mesh.getWorldPosition(new THREE.Vector3())
      .distanceTo(target) <= Math.max(.06, sigma*3));
    if (candidates.length !== 1) throw new Error('视觉抓取点未对应唯一可移动物体');
    return candidates[0];
  }
  attach(item: Prop, hand: THREE.Object3D, palm: THREE.Vector3): void {
    const box = new THREE.Box3().setFromObject(item.mesh);
    if (box.distanceToPoint(palm) > .045) throw new Error('手尚未接触物体，不能建立持有约束');
    hand.attach(item.mesh); item.held = true;
  }
  release(item: Prop): void { this.group.attach(item.mesh); item.held = false; }
  consume(item: Prop, operation: 'eat' | 'drink'): number {
    if (item.held || item.mesh.getWorldPosition(new THREE.Vector3()).distanceTo(item.home) > .065)
      throw new Error('吃喝完成前必须将物品放回支撑面');
    if (item.definition.kind !== (operation === 'eat' ? 'food' : 'water') || item.remaining <= 0)
      throw new Error('目标不是可用的食物或饮水');
    item.remaining = Math.max(0, item.remaining-1/(item.definition.portions ?? 2));
    if (operation === 'eat') item.mesh.scale.setScalar(.65+.35*item.remaining);
    const water = item.mesh.getObjectByName('water');
    if (water) water.position.y = item.definition.size[1]*(-.35+.67*item.remaining);
    item.mesh.visible = item.remaining > 0;
    return item.remaining;
  }
  reset(): void {
    for (const item of this.items) { this.release(item); item.mesh.position.copy(item.home); item.mesh.quaternion.identity();
      item.remaining = 1; item.mesh.visible = true; item.mesh.scale.setScalar(1);
      const water = item.mesh.getObjectByName('water'); if (water) water.position.y = item.definition.size[1]*.32; }
  }
  dispose(): void {
    this.reset();
    this.group.traverse(node => { if (node instanceof THREE.Mesh) {
      node.geometry.dispose(); const materials = Array.isArray(node.material) ? node.material : [node.material]; materials.forEach(material => material.dispose());
    } });
    this.group.removeFromParent(); this.items = [];
  }
}
export type HeldProp = ReturnType<InteractionProps['match']>;

import * as THREE from 'three';
import type { PropDefinition } from './interaction_props';

function solid(geometry: THREE.BufferGeometry, color: number, roughness = .65): THREE.Mesh {
  const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color, roughness }));
  mesh.castShadow = true; mesh.receiveShadow = true; return mesh;
}

export function sandwichMesh(size: number[]): THREE.Group {
  const group = new THREE.Group(), [width, height, depth] = size;
  const shape = new THREE.Shape(); shape.moveTo(-width/2, -depth/2); shape.lineTo(width/2, -depth/2);
  shape.lineTo(-width/2, depth/2); shape.closePath();
  const layers = [[.26, 0xe4bb78], [.07, 0x77a843], [.12, 0xd98071], [.09, 0xf0d46b], [.07, 0x77a843], [.26, 0xe4bb78]];
  let y = -height/2;
  for (const [fraction, color] of layers) {
    const thickness = fraction*height;
    const mesh = solid(new THREE.ExtrudeGeometry(shape, { depth: thickness, bevelEnabled: false }), color);
    mesh.rotation.x = -Math.PI/2; mesh.position.y = y; group.add(mesh); y += thickness;
  }
  return group;
}

export function cupMesh(size: number[]): THREE.Group {
  const group = new THREE.Group(), radius = size[0]/2, height = size[1];
  const profile = [[0, -height/2], [radius*.84, -height/2], [radius, height/2],
    [radius-.004, height/2], [radius*.84-.004, -height/2+.006]];
  group.add(solid(new THREE.LatheGeometry(profile.map(([x, y]) => new THREE.Vector2(x, y)), 28), 0xf1efe8, .22));
  const handle = solid(new THREE.TorusGeometry(height*.23, .006, 8, 20), 0xf1efe8, .22);
  handle.position.x = radius; group.add(handle);
  const water = solid(new THREE.CylinderGeometry(radius*.88, radius*.88, .004, 24), 0x79b5c7, .15);
  water.name = 'water'; water.position.y = height*.32; group.add(water);
  const stripe = solid(new THREE.CylinderGeometry(radius*.95, radius*.93, height*.16, 28, 1, true), 0x356b86);
  stripe.position.y = height*.12; group.add(stripe);
  return group;
}

export function propMesh(definition: PropDefinition): THREE.Group | null {
  if (definition.kind === 'food') return sandwichMesh(definition.size);
  if (definition.kind === 'water') return cupMesh(definition.size);
  return null;
}

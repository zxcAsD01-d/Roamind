import * as THREE from 'three';

const SIZE = 1024;
const METERS = 8;

function randomSource(): () => number {
  let value = 0x68ab23;
  return () => {
    value = (Math.imul(value, 1664525)+1013904223) >>> 0;
    return value/4294967296;
  };
}

/** Multi-scale plaster aggregate: static visual detail, with no navigation data. */
function plasterMap(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = SIZE;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('无法创建墙面灰泥纹理');
  const random = randomSource();
  context.fillStyle = '#e8e2d6'; context.fillRect(0, 0, SIZE, SIZE);
  for (const [count, low, high, opacity] of [[900, 8, 35, .035], [12000, 1, 6, .10], [4000, .5, 1.5, .10]]) {
    for (let index = 0; index < count; index++) {
      const x = random()*SIZE, y = random()*SIZE, radius = low+random()*(high-low);
      context.fillStyle = random() < .5 ? 'rgba(83,77,64,'+opacity+')' : 'rgba(255,253,240,'+opacity+')';
      context.beginPath(); context.ellipse(x, y, radius, radius*(.4+random()*.6), random()*Math.PI, 0, Math.PI*2); context.fill();
    }
  }
  const map = new THREE.CanvasTexture(canvas);
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = map.wrapT = THREE.RepeatWrapping;
  map.anisotropy = 4;
  return map;
}

/** World-metre UVs retain batching and avoid stretching a texture over long walls. */
export function wallMaterial(): THREE.MeshStandardMaterial {
  const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: .93, map: plasterMap() });
  return worldSurfaceMaterial(material, METERS);
}

export function worldSurfaceMaterial(material: THREE.MeshStandardMaterial, meters: number): THREE.MeshStandardMaterial {
  material.onBeforeCompile = shader => {
    const uv = [
      '#include <uv_vertex>',
      '#ifdef USE_MAP',
      'vec4 plasterPoint = vec4(position, 1.0);',
      'vec3 plasterNormal = normal;',
      '#ifdef USE_INSTANCING',
      'plasterPoint = instanceMatrix * plasterPoint;',
      'plasterNormal = mat3(instanceMatrix) * plasterNormal;',
      '#endif',
      'plasterPoint = modelMatrix * plasterPoint;',
      'vec3 side = abs(plasterNormal);',
      'vec2 plasterUv = side.y > max(side.x, side.z) ? plasterPoint.xz',
      '  : (side.x > side.z ? plasterPoint.zy : plasterPoint.xy);',
      'vMapUv = plasterUv / '+meters.toFixed(2)+';',
      '#endif',
    ].join('\n');
    if (!shader.vertexShader.includes('#include <uv_vertex>')) throw new Error('墙面着色器缺少 UV 接入点');
    shader.vertexShader = shader.vertexShader.replace('#include <uv_vertex>', uv);
  };
  material.customProgramCacheKey = () => 'world-metre-surface-v2-'+meters;
  return material;
}

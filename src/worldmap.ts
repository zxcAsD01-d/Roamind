import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { SemanticObject, StereoResult, SurfaceSegment, VolumeSegment, WorldCloud } from './stereo';

/** Reconstructed observations only; never reads the simulation scene. */
export class VisualWorldMap {
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(50, 320/230, .02, 80);
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private geometry = new THREE.BufferGeometry();
  private positions = this.attribute(6000);
  private colors = this.attribute(6000);
  private mapRevision = -1;
  private mapId: string | undefined;
  private geometryRevision = -1;
  private retiredMaps = new Set<string>();
  private cloud = new THREE.Points(this.geometry, new THREE.PointsMaterial({ size: .035, vertexColors: true }));
  private surfaceGeometry = new THREE.BufferGeometry();
  private surfacePositions = this.attribute(6144);
  private surfaceColors = this.attribute(6144);
  private surfaces = new THREE.Mesh(this.surfaceGeometry,
    new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true,
      polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
  private volumeGeometry = new THREE.BufferGeometry();
  private volumePositions = this.attribute(512*24);
  private volumeColors = this.attribute(512*24);
  private volumes = new THREE.LineSegments(this.volumeGeometry, new THREE.LineBasicMaterial({ vertexColors: true }));
  private trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xe9cb76 }));
  private marker = new THREE.AxesHelper(.35);
  private framed = false;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false });
    this.renderer.setSize(320, 230, false); this.renderer.setPixelRatio(1);
    this.bindAttributes(this.geometry, this.positions, this.colors);
    this.bindAttributes(this.surfaceGeometry, this.surfacePositions, this.surfaceColors);
    this.bindAttributes(this.volumeGeometry, this.volumePositions, this.volumeColors);
    this.surfaces.frustumCulled = false; this.volumes.frustumCulled = false;
    this.scene.background = new THREE.Color(0x111a1b);
    this.scene.add(this.cloud, this.surfaces, this.trail, this.marker, this.volumes);
    this.camera.position.set(3, 3, 3);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.addEventListener('change', () => this.render());
    this.render();
  }

  private attribute(vertices: number): THREE.BufferAttribute {
    return new THREE.BufferAttribute(new Float32Array(vertices*3), 3).setUsage(THREE.DynamicDrawUsage);
  }

  private bindAttributes(geometry: THREE.BufferGeometry, positions: THREE.BufferAttribute,
    colors: THREE.BufferAttribute): void {
    geometry.setAttribute('position', positions).setAttribute('color', colors).setDrawRange(0, 0);
  }

  update(result: StereoResult): void {
    this.updateCloud(result.map);
    this.trail.geometry.dispose();
    this.trail.geometry = new THREE.BufferGeometry().setFromPoints(result.trajectory.map(p => new THREE.Vector3().fromArray(p)));
    const matrix = new THREE.Matrix4().set(...result.pose.cameraToWorld as Parameters<THREE.Matrix4['set']>);
    this.marker.matrixAutoUpdate = false; this.marker.matrix.copy(matrix);
    this.marker.visible = true;
    this.render();
  }

  updateMap(map: WorldCloud): void {
    this.updateCloud(map);
    this.render();
  }

  private acceptMap(map: WorldCloud): boolean {
    if (map.mapId && this.retiredMaps.has(map.mapId)) return false;
    if (map.mapId !== this.mapId) {
      if (this.mapId) this.retiredMaps.add(this.mapId);
      if (this.retiredMaps.size > 16) this.retiredMaps.delete(this.retiredMaps.values().next().value!);
      this.mapId = map.mapId; this.mapRevision = -1; this.geometryRevision = -1; this.framed = false;
    }
    return map.revision > this.mapRevision;
  }

  private updateCloud(map: WorldCloud): void {
    if (!this.acceptMap(map)) return;
    const count = Math.min(map.colors.length, this.positions.count);
    const color = new THREE.Color(), point = new THREE.Vector3(), bounds = new THREE.Box3();
    for (let index = 0; index < count; index++) {
      this.positions.setXYZ(index, map.positions[index*3], map.positions[index*3+1], map.positions[index*3+2]);
      color.setHex(map.colors[index], THREE.SRGBColorSpace);
      if (map.inferred?.[index]) color.multiplyScalar(.65);
      this.colors.setXYZ(index, color.r, color.g, color.b);
      bounds.expandByPoint(point.fromBufferAttribute(this.positions, index));
    }
    this.upload(this.geometry, this.positions, this.colors, count);
    this.geometry.boundingBox = bounds;
    this.geometry.boundingSphere = bounds.getBoundingSphere(new THREE.Sphere());
    this.mapRevision = map.revision;
    const revision = map.geometryRevision ?? map.revision;
    if (revision !== this.geometryRevision) {
      this.updateSurfaces(map.surfaces ?? []);
      this.updateVolumes(map.volumes ?? []);
      this.geometryRevision = revision;
    }
    this.updateObjects(map.objects);
    if (!this.framed && count > 100) this.frame();
  }

  private upload(geometry: THREE.BufferGeometry, positions: THREE.BufferAttribute,
    colors: THREE.BufferAttribute, count: number): void {
    positions.clearUpdateRanges(); colors.clearUpdateRanges();
    if (count) { positions.addUpdateRange(0, count*3); colors.addUpdateRange(0, count*3); }
    positions.needsUpdate = true; colors.needsUpdate = true;
    geometry.setDrawRange(0, count);
  }

  private updateSurfaces(surfaces: SurfaceSegment[]): void {
    const needed = Math.min(36000, surfaces.reduce((sum, surface) => sum+surface.indices.length, 0));
    if (needed > this.surfacePositions.count) {
      this.surfaceGeometry.dispose();
      const capacity = Math.min(36000, Math.max(needed, this.surfacePositions.count*2));
      this.surfacePositions = this.attribute(capacity); this.surfaceColors = this.attribute(capacity);
      this.bindAttributes(this.surfaceGeometry, this.surfacePositions, this.surfaceColors);
    }
    let offset = 0;
    const color = new THREE.Color();
    for (const surface of surfaces) {
      color.setHex(surface.color, THREE.SRGBColorSpace);
      if (surface.measuredRatio < .5) color.multiplyScalar(.65);
      for (const vertex of surface.indices) {
        if (offset >= needed) break;
        this.surfacePositions.setXYZ(offset, surface.vertices[vertex*3], surface.vertices[vertex*3+1], surface.vertices[vertex*3+2]);
        this.surfaceColors.setXYZ(offset++, color.r, color.g, color.b);
      }
    }
    this.upload(this.surfaceGeometry, this.surfacePositions, this.surfaceColors, offset);
  }

  private updateVolumes(volumes: VolumeSegment[]): void {
    const edges = [0, 1, 0, 2, 0, 4, 1, 3, 1, 5, 2, 3, 2, 6, 3, 7, 4, 5, 4, 6, 5, 7, 6, 7];
    const color = new THREE.Color();
    let offset = 0;
    for (const volume of volumes.slice(0, 512)) {
      if (volume.semantic) color.setHSL((volume.semantic*.618)%1, .7, .6);
      else color.setHex(0x718080);
      for (const corner of edges) {
        this.volumePositions.setXYZ(offset, volume.bounds[corner & 4 ? 1 : 0][0],
          volume.bounds[corner & 2 ? 1 : 0][1], volume.bounds[corner & 1 ? 1 : 0][2]);
        this.volumeColors.setXYZ(offset++, color.r, color.g, color.b);
      }
    }
    this.upload(this.volumeGeometry, this.volumePositions, this.volumeColors, offset);
  }

  private frame(): void {
    this.framed = true;
    const box = new THREE.Box3().setFromObject(this.cloud);
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(2, Math.min(8, box.getSize(new THREE.Vector3()).length()*.7));
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(radius*.6, radius, radius));
    this.controls.update();
  }

  private updateObjects(objects: SemanticObject[]): void {
    const list = document.querySelector('#world-objects');
    if (list) list.replaceChildren(...objects.map(object => {
      const item = document.createElement('li');
      item.textContent = `${object.label} · ${object.center.map(v => v.toFixed(1)).join(', ')} m · ${object.visible ? '可见' : `记忆 ${object.ageSeconds.toFixed(0)}s`}`;
      return item;
    }));
  }

  resetView(): void {
    this.framed = false; this.mapRevision = -1; this.mapId = undefined; this.geometryRevision = -1;
    this.retiredMaps.clear();
    this.geometry.setDrawRange(0, 0); this.surfaceGeometry.setDrawRange(0, 0); this.volumeGeometry.setDrawRange(0, 0);
    this.geometry.boundingBox = new THREE.Box3(); this.geometry.boundingSphere = new THREE.Sphere();
    this.trail.geometry.dispose(); this.trail.geometry = new THREE.BufferGeometry();
    this.marker.visible = false;
    this.updateObjects([]);
    this.render();
  }

  private render(): void { this.renderer.render(this.scene, this.camera); }
}

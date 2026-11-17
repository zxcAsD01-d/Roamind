import * as THREE from 'three';

export type ComputerDefinition = { id: string; screen: number[]; screenSize: number[]; keyboard: number[]; mouse: number[] };
export type ComputerState = { id: string; active: boolean; elapsed: number; inputs: number; score: number; completed: boolean };
type Key = 'left' | 'right' | 'space';

/** In-world keyboard game. The controller reads rendered pixels, then submits actual key presses. */
export class ComputerStation {
  readonly group = new THREE.Group();
  private canvas = document.createElement('canvas');
  private ctx: CanvasRenderingContext2D;
  private texture: THREE.CanvasTexture;
  private time = 0;
  private paddle = 240;
  private target = { x: 150, y: 85 };
  private cooldown = 0;
  private state: ComputerState;

  constructor(readonly definition: ComputerDefinition) {
    this.canvas.width = 480; this.canvas.height = 270;
    const context = this.canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('无法创建电脑屏幕');
    this.ctx = context;
    this.state = this.initial();
    this.texture = new THREE.CanvasTexture(this.canvas); this.texture.colorSpace = THREE.SRGBColorSpace;
    const screen = new THREE.Mesh(new THREE.PlaneGeometry(...definition.screenSize as [number, number]),
      new THREE.MeshBasicMaterial({ map: this.texture, toneMapped: false }));
    screen.position.fromArray(definition.screen); this.group.add(screen);
    this.addKeyboard(); this.render();
  }

  private initial(): ComputerState { return { id: this.definition.id, active: false, elapsed: 0, inputs: 0, score: 0, completed: false }; }
  snapshot(): ComputerState { return { ...this.state }; }
  start(): void { this.state = { ...this.initial(), active: true }; this.time = 0; this.paddle = 240; this.target = { x: 150, y: 85 }; this.cooldown = 0; this.render(); }
  stop(completed = false): void { this.state.active = false; this.state.completed = completed; this.render(); }
  reset(): void { this.state = this.initial(); this.render(); }

  private addKeyboard(): void {
    const material = new THREE.MeshStandardMaterial({ color: 0xd4dce2, roughness: .5 });
    const geometry = new THREE.BoxGeometry(.022, .004, .022);
    const keys = new THREE.InstancedMesh(geometry, material, 45), matrix = new THREE.Matrix4();
    for (let row = 0; row < 3; row++) for (let col = 0; col < 15; col++) {
      matrix.makeTranslation(this.definition.keyboard[0]+.06+(col-7)*.027, this.definition.keyboard[1]+.006, this.definition.keyboard[2]+(row-1)*.035);
      keys.setMatrixAt(row*15+col, matrix);
    }
    this.group.add(keys);
  }

  /** Confirm the measured keyboard against a unique physical device at execution time. */
  matches(target: THREE.Vector3, sigma: number): boolean {
    return target.distanceTo(new THREE.Vector3().fromArray(this.definition.keyboard)) <= .12+sigma;
  }

  contact(palms: THREE.Vector3[]): boolean {
    const center = new THREE.Vector3().fromArray(this.definition.keyboard);
    return palms.length === 2 && palms.every(p => Math.abs(p.x-center.x-.06) < .24
      && Math.abs(p.z-center.z) < .13 && Math.abs(p.y-center.y) < .045);
  }

  private observeKey(): Key {
    const { data, width, height } = this.ctx.getImageData(0, 60, 480, 185);
    let sum = 0, count = 0;
    for (let y = 0; y < height; y += 2) for (let x = 0; x < width; x += 2) {
      const i = (y*width+x)*4;
      if (data[i] < 90 && data[i+1] > 200 && data[i+2] > 200) { sum += x; count++; }
    }
    if (!count) return 'space';
    const delta = sum/count-this.paddle;
    return Math.abs(delta) < 12 ? 'space' : delta < 0 ? 'left' : 'right';
  }

  update(dt: number, touching: boolean): Key | null {
    if (!this.state.active) return null;
    const step = Math.max(0, Math.min(.05, dt)); this.time += step; this.state.elapsed += step; this.cooldown -= step;
    let key: Key | null = null;
    if (touching && this.cooldown <= 0) {
      key = this.observeKey(); this.state.inputs++; this.cooldown = .12;
      this.paddle = THREE.MathUtils.clamp(this.paddle+(key === 'left' ? -18 : key === 'right' ? 18 : 0), 30, 450);
    }
    this.target.y += step*64;
    if (this.target.y >= 226) {
      if (Math.abs(this.target.x-this.paddle) < 34) this.state.score++;
      this.target = { x: 55+(Math.sin(this.time*1.71)*.5+.5)*370, y: 76 };
    }
    this.render(); return key;
  }

  private render(): void {
    const c = this.ctx;
    c.fillStyle = '#101b2a'; c.fillRect(0, 0, 480, 270);
    c.fillStyle = '#f1eee6'; c.font = 'bold 17px sans-serif'; c.fillText('ORBIT  /  PLAY ROOM', 22, 31);
    c.font = '12px sans-serif'; c.fillStyle = '#99acbf';
    c.fillText(`SCORE  ${String(this.state.score).padStart(2, '0')}       ${this.state.elapsed.toFixed(0)}s`, 22, 51);
    if (this.state.active) {
      c.strokeStyle = '#23344a'; for (let x = 22; x < 480; x += 36) { c.beginPath(); c.moveTo(x, 65); c.lineTo(x, 245); c.stroke(); }
      c.fillStyle = '#38efed'; c.beginPath(); c.arc(this.target.x, this.target.y, 10, 0, Math.PI*2); c.fill();
      c.fillStyle = '#ffcf85'; c.fillRect(this.paddle-30, 231, 60, 7);
    } else {
      c.fillStyle = '#e8bd7f'; c.font = 'bold 24px sans-serif'; c.fillText(this.state.completed ? 'NICE SESSION!' : 'CATCH THE SIGNAL', 75, 137);
      c.fillStyle = '#a3b5c8'; c.font = '14px sans-serif'; c.fillText('Sit down. Find your rhythm.', 135, 170);
    }
    c.fillStyle = '#99acbf'; c.font = '11px sans-serif'; c.fillText('←  →  MOVE      SPACE  HOLD', 22, 260);
    this.texture.needsUpdate = true;
  }

  dispose(): void {
    this.group.traverse(node => { if (node instanceof THREE.Mesh) {
      node.geometry.dispose(); for (const material of Array.isArray(node.material) ? node.material : [node.material]) material.dispose();
      if (node instanceof THREE.InstancedMesh) node.dispose();
    } });
    this.texture.dispose(); this.group.removeFromParent();
  }
}

export function matchComputer(stations: ComputerStation[], point: THREE.Vector3, sigma: number): ComputerStation {
  const found = stations.filter(station => station.matches(point, sigma));
  if (found.length !== 1) throw new Error('视觉键盘未匹配到唯一的可操作电脑');
  return found[0];
}

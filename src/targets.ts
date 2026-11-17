import * as THREE from 'three';
import type { ObservedTarget } from './perception';

export const targetPanel = `
  <details class="target-panel" open><summary>面向 / 动态跟随</summary>
    <label class="field-label" for="facing-angle">绝对面向角度（0=+Z，90=+X）</label>
    <input id="facing-angle" class="scene-input" type="number" value="0" step="15">
    <div class="scene-buttons"><button id="face-heading" class="secondary-button">转向该角度</button><label class="loop-control"><input id="move-facing" type="checkbox"><span>移动时保持此面向</span></label></div>
    <label class="setting-row"><span>仿真观测目标（双击地面移动）</span><input id="target-toggle" type="checkbox"><i class="switch"></i></label>
    <label class="field-label" for="target-select">观测目标</label><select id="target-select"><option value="">暂无新鲜观测</option></select>
    <label class="field-label" for="follow-distance">跟随距离 / 米</label><input id="follow-distance" class="scene-input" type="number" min="0.65" max="5" step="0.1" value="1.2">
    <div class="scene-buttons"><button id="face-target" class="secondary-button">面向目标</button><button id="move-target" class="secondary-button">移到观测位置附近</button></div>
    <div class="scene-buttons"><button id="follow-target" class="primary-button">持续跟随</button><button id="stop-follow" class="secondary-button">停止跟随</button></div>
    <p id="tracking-status" class="ardy-status">只读最新观测 · 丢失即停 · 2秒滚动规划</p>
  </details>`;

export class TargetMarkers {
  private marker = new THREE.Mesh(new THREE.SphereGeometry(.20,16,12),new THREE.MeshStandardMaterial({color:0x81bdf1,emissive:0x123849}));
  private lastPublish=0;
  private enabled=false;
  constructor(scene: THREE.Scene, canvas: HTMLCanvasElement, camera: THREE.Camera,
              private observe:(body:object)=>Promise<object>, private free:(point:THREE.Vector3)=>boolean) {
    this.marker.position.set(-5,.5,4.3); this.marker.visible=false; scene.add(this.marker);
    document.querySelector('#target-toggle')!.addEventListener('change',event=>{
      this.enabled=(event.target as HTMLInputElement).checked; this.marker.visible=this.enabled;
    });
    canvas.addEventListener('dblclick',event=>{
      if(!this.enabled) return;
      const rect=canvas.getBoundingClientRect(); const ray=new THREE.Raycaster();
      ray.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,1-(event.clientY-rect.top)/rect.height*2),camera);
      const hit=ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,1,0),0),new THREE.Vector3());
      if(hit && this.free(hit)) {this.marker.position.set(hit.x,.5,hit.z); this.lastPublish=0;}
    });
  }
  update():void {
    if(!this.enabled || performance.now()-this.lastPublish<300) return;
    this.lastPublish=performance.now();
    void this.observe({id:'sim-target',label:'蓝色仿真目标',source:'simulation',position:[this.marker.position.x,.92,this.marker.position.z],
      ttl:1.5,observedAt:Date.now()/1000}).catch(error=>{document.querySelector('#tracking-status')!.textContent=error instanceof Error?error.message:'目标发布失败';});
  }
}

export function renderTargets(targets:ObservedTarget[]):void {
  const select=document.querySelector<HTMLSelectElement>('#target-select')!;
  const selected=select.value;
  select.replaceChildren(...targets.filter(target=>!target.stale).map(target=>{
    const option=document.createElement('option');option.value=target.id;option.textContent=`${target.label} · ${target.source}${target.position?'':' · 仅方向'}`;return option;
  }));
  if([...select.options].some(option=>option.value===selected)) select.value=selected;
}

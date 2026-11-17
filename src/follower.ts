import type { ObservedTarget } from './perception';

export type FollowOptions = { targetId: string; distance?: number; speed?: number; faceTarget?: boolean };
type Playback = { buffered: number; playing: boolean; generating: boolean };
type FollowHooks = { targets:()=>ObservedTarget[]; playback:()=>Playback;
  generate:(options:FollowOptions, stillCurrent:()=>boolean)=>Promise<void>; hold:()=>void;
  status:(message:string)=>void; failed:(message:string)=>void };

/** Serial, bounded receding-horizon control; never queues a target's future. */
export class TargetFollower {
  private revision=0;
  private options:FollowOptions|null=null;
  private lastPosition:number[]|null=null;
  private lastPlan=0;
  private lostAt=0;
  constructor(private hooks:FollowHooks) {}
  get active():boolean {return this.options!==null;}
  start(options:FollowOptions):void {
    this.stop(); this.options=options; this.lastPosition=null; this.lastPlan=0;
    void this.tick(this.revision);
  }
  stop():void {
    const active=this.active;
    this.revision++;this.options=null;this.lostAt=0;
    if(active) this.hooks.status('跟随已停止');
  }

  private async tick(revision:number):Promise<void> {
    const options=this.options;
    if(!options || revision!==this.revision) return;
    try {
      const target=this.hooks.targets().find(t=>t.id===options.targetId);
      if(!target || target.stale || Date.now()/1000-target.observedAt>target.ttl || !target.position) {
        if(!this.lostAt) {this.lostAt=performance.now(); this.hooks.hold();}
        this.hooks.status('目标丢失：保持当前姿势，等待新观测');
        if(performance.now()-this.lostAt>10000) throw new Error('目标持续丢失10秒，跟随已结束');
      } else {
        this.lostAt=0;
        const state=this.hooks.playback();
        const changed=this.lastPosition && Math.hypot(target.position[0]-this.lastPosition[0],target.position[2]-this.lastPosition[2])>.6;
        if(((state.buffered<.85 || !state.playing) && !state.generating) || (changed && performance.now()-this.lastPlan>1200)) {
          this.lastPosition=target.position.slice(); this.lastPlan=performance.now();
          this.hooks.status(`跟随 ${target.label} · 当前观测 · 2秒窗口`);
          await this.hooks.generate(options,()=>revision===this.revision);
        }
      }
    } catch(error) {
      if(revision===this.revision) {this.stop();this.hooks.hold();this.hooks.failed(error instanceof Error?error.message:'跟随失败');}
    } finally {
      if(revision===this.revision) window.setTimeout(()=>{void this.tick(revision);},200);
    }
  }
}

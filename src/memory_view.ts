import type { BehaviorMemory, SpatialTopology } from './character_types';

const REGION_COLORS = ['#477d62', '#456b88', '#756487', '#7d7547', '#437e7d', '#875e58'];
const BOUNDARY_COLORS = { supported: '#d09c8c', uncertain: '#ac9981', unknown: '#64716d', opening: '#74c7ed' };

export class MemoryView {
  private bounds = { x: 0, z: 0, scale: 1 };
  private lastMemoryFrame = '';

  private describe(memory: BehaviorMemory['spatialMemory']): void {
    const age = memory.observedAt ? Date.now()/1000-memory.observedAt : Infinity;
    document.querySelector('#character-memory')!.textContent = `定位 ${memory.position?.map(v => v.toFixed(2)).join(', ') ?? '未知'} · ${age < 1.5 ? '新鲜观测' : '历史记忆'}\n开放 ${memory.cells.open} / 障碍 ${memory.cells.closed} / 待确认 ${memory.cells.uncertain} · 访问 ${memory.visitedRegions} 格 · 人物 ${memory.peopleRegions.length}`;
    if (memory.searchCoverage) {
      const coverage = memory.searchCoverage;
      document.querySelector('#character-memory')!.textContent += `\n方向已确认 ${coverage.regions.filter(region => region.confirmed).length} 区 · 出口候选 ${coverage.exits.filter(exit => exit.status === 'unvisited').length} / 已看 ${coverage.exits.filter(exit => exit.lookedAt).length}（不代表封闭）`;
    }
    const topology = memory.topology, floor = memory.floor;
    if (floor) {
      const reliable = floor.tiltDeg <= 12 && floor.support >= 80 && floor.residual <= .04;
      document.querySelector('#character-memory')!.textContent += reliable
        ? `\n地板候选 ${floor.tiltDeg.toFixed(1)}° / 支撑 ${floor.support} 点`
        : `\n地板拟合待复核（候选倾角 ${floor.tiltDeg.toFixed(1)}°）`;
    }
    if (topology) {
      const region = topology.regions.find(item => item.id === topology.currentRegion);
      const boundary = region?.enclosure?.closedBoundary ? '已观测地板边界闭合' :
        region?.enclosure?.status === 'connected' ? '与其他区域联通' :
        region?.boundary === 'observed-open' ? '有实测开口，通行待确认' :
        region?.boundary === 'uncertain-boundary' ? '边界置信不足，继续探查' : '仍有开放未知边界';
      document.querySelector('#character-memory')!.textContent += `\n区域 ${topology.currentRegion ?? '未知'} · ${region?.personChecked ? '已环顾查人' : '待环顾'} · ${boundary} · 门 ${topology.doors.length} / 通口 ${topology.passages.length}`;
      if (topology.exits) {
        const exits = topology.exits.filter(exit => exit.regions.includes(topology.currentRegion ?? -1));
        document.querySelector('#character-memory')!.textContent += `\n本区出口 ${exits.filter(exit => exit.traversable).length} 可通行 / ${exits.filter(exit => exit.state === 'blocked').length} 受阻 / ${exits.filter(exit => !exit.traversable && exit.state !== 'blocked').length} 待确认 · 联通 ${region?.reachableRegionCount ?? 0} 区`;
        const space = topology.spaces?.find(item => item.id === region?.componentId);
        if (space?.enclosure.closedBoundary && space.regionIds.length > 1) {
          document.querySelector('#character-memory')!.textContent += '\n联通片区的已观测地板边界闭合';
        }
      }
    }
    if (memory.exitMemory) {
      const portals = memory.exitMemory.portals;
      const confidence = portals.length ? Math.round(100*Math.max(...portals.map(item => item.confidence))) : 0;
      document.querySelector('#character-memory')!.textContent += `\n点云通口 ${portals.length} · 最高置信 ${confidence}% · 已穿越 ${memory.exitMemory.crossings.length} 次`;
    }
  }

  update(memory: BehaviorMemory['spatialMemory']): void {
    const canvas = document.querySelector<HTMLCanvasElement>('#memory-grid')!;
    if (!canvas.getClientRects().length) return;
    this.describe(memory);
    const signature = `${memory.observedAt}:${memory.topology?.revision ?? 0}`;
    if (signature === this.lastMemoryFrame) return;
    this.lastMemoryFrame = signature;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('无法创建记忆地图');
    ctx.fillStyle = '#111a1b'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    const cells = memory.knownCells;
    if (!cells.length) return;
    const xs = cells.map(c => c[0]*memory.resolution), zs = cells.map(c => c[1]*memory.resolution);
    const minX = Math.min(...xs)-.7, maxX = Math.max(...xs)+1;
    const minZ = Math.min(...zs)-.7, maxZ = Math.max(...zs)+1;
    const scale = Math.min(canvas.width/(maxX-minX), canvas.height/(maxZ-minZ));
    this.bounds = { x: minX, z: minZ, scale };
    for (const [x, z, kind, region] of cells) {
      ctx.fillStyle = kind < 0 ? REGION_COLORS[(region ?? 0)%REGION_COLORS.length]! : kind > 0 ? '#b26659' : '#535b59';
      ctx.fillRect((x*memory.resolution-minX)*scale, (z*memory.resolution-minZ)*scale,
        Math.max(1, memory.resolution*scale-1), Math.max(1, memory.resolution*scale-1));
    }
    if (memory.topology) this.drawTopology(ctx, memory.topology);
    const dot = (position: number[], color: string): void => {
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc((position[0]-minX)*scale, (position[2]-minZ)*scale, 4, 0, Math.PI*2); ctx.fill();
    };
    memory.peopleRegions.forEach(person => dot(person.position, '#e8c16e'));
    memory.searchCoverage?.exits.forEach(exit => dot(exit.position,
      exit.status === 'unvisited' ? '#74c7ed' : exit.status === 'inspected' ? '#667875' : '#b26659'));
    if (!memory.topology?.exits) {
      memory.topology?.doors.forEach(door => dot(door.position, door.traversable ? '#f3a754' : '#a47355'));
      memory.exitMemory?.portals.forEach(portal => dot(portal.point, portal.confidence >= .75 ? '#c6a0f6' : '#82758d'));
    }
    if (memory.position) dot(memory.position, '#ffffff');
  }

  private drawTopology(ctx: CanvasRenderingContext2D, topology: SpatialTopology): void {
    const { x, z, scale } = this.bounds;
    const line = (a: number[], b: number[]): void => {
      ctx.beginPath(); ctx.moveTo((a[0]-x)*scale, (a[2]-z)*scale);
      ctx.lineTo((b[0]-x)*scale, (b[2]-z)*scale); ctx.stroke();
    };
    for (const boundary of topology.boundarySegments ?? []) {
      ctx.strokeStyle = BOUNDARY_COLORS[boundary.state]; ctx.lineWidth = boundary.state === 'supported' ? 2 : 1;
      ctx.setLineDash(boundary.state === 'unknown' ? [2, 3] : []);
      line(boundary.start, boundary.end);
    }
    ctx.setLineDash([]);
    for (const exit of topology.exits ?? []) {
      ctx.strokeStyle = exit.traversable ? '#91d3be' : '#b26659';
      if (exit.connectsRegions) {
        ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
        for (const identity of exit.regions) {
          const region = topology.regions.find(item => item.id === identity);
          if (region) line(exit.position, region.center);
        }
        ctx.setLineDash([]);
      }
      ctx.fillStyle = exit.state === 'blocked' ? '#e98576' : exit.traversable ? '#91d3be' : '#f3c577';
      ctx.beginPath(); ctx.arc((exit.position[0]-x)*scale, (exit.position[2]-z)*scale, 4, 0, Math.PI*2); ctx.fill();
    }
  }

  positionAt(event: MouseEvent): [number, number] {
    const canvas = event.currentTarget as HTMLCanvasElement, rect = canvas.getBoundingClientRect();
    return [this.bounds.x+(event.clientX-rect.left)*canvas.width/rect.width/this.bounds.scale,
      this.bounds.z+(event.clientY-rect.top)*canvas.height/rect.height/this.bounds.scale];
  }
}

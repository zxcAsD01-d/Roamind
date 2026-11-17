import type { LumenAction, LumenEntry, LumenManifest, LumenSource } from './types';

const ROOT = '/external/HAST-SEC/';
const ACTION_KEYS: Record<string, string> = {
  wave: 'right-摆动手臂', raise: 'right-向上举过头部', point: 'right-向前移动',
};
export async function lumenJson<T>(path: string): Promise<T> {
  const response = await fetch(ROOT+path);
  if (!response.ok) throw new Error('流明资源读取失败 '+response.status+': '+path);
  return await response.json() as T;
}

/** Original resources stay in the vendor tree, including their license. */
export class LumenAssets {
  private sources = new Map<string, Float32Array>();
  private pendingSources = new Map<string, Promise<Float32Array>>();
  private entries = new Map<string, LumenEntry>();
  private descriptors = new Map<string, LumenSource>();
  private constructor(readonly manifest: LumenManifest, sources: LumenSource[]) {
    for (const source of sources) this.descriptors.set(source.id, source);
  }

  static async load(): Promise<LumenAssets> {
    const [manifest, ...sources] = await Promise.all([
      lumenJson<LumenManifest>('models/skills/manifest.json'),
      ...[0, 1, 2, 3].map(i => lumenJson<LumenSource[]>('models/skills/sources-'+i+'.json')),
    ] as const);
    if (manifest.rate !== 20 || manifest.bones.length !== 53 || !Array.isArray(manifest.classes))
      throw new Error('流明骨架或采样率不兼容');
    return new LumenAssets(manifest, sources.flat());
  }

  async entry(action: LumenAction): Promise<LumenEntry> {
    const key = ACTION_KEYS[action] ?? action.slice('lumen:'.length), cached = this.entries.get(key);
    if (cached) return cached;
    const descriptor = this.manifest.classes.find(item => item.key === key);
    if (!descriptor || !/^class-[0-9]+[.]json$/.test(descriptor.file)) throw new Error('流明动作不存在: '+action);
    const entry = await lumenJson<LumenEntry>('models/skills/'+descriptor.file);
    if (entry.key !== key || !entry.skills?.length || entry.mask.some(i => !Number.isInteger(i)
      || !/^(left|right)(Shoulder|UpperArm|LowerArm|Hand)$/.test(this.manifest.bones[i] ?? '')))
      throw new Error('流明上身动作遮罩无效');
    this.entries.set(key, entry);
    return entry;
  }

  async source(identity: string): Promise<Float32Array> {
    const cached = this.sources.get(identity);
    if (cached) return cached;
    const pending = this.pendingSources.get(identity);
    if (pending) return pending;
    const loading = this.loadSource(identity);
    this.pendingSources.set(identity, loading);
    try { return await loading; }
    finally { this.pendingSources.delete(identity); }
  }

  private async loadSource(identity: string): Promise<Float32Array> {
    const descriptor = this.descriptors.get(identity);
    if (!descriptor || !/^[A-Za-z0-9_-]+[.]bin$/.test(descriptor.file)) throw new Error('流明动作来源无效');
    const response = await fetch(ROOT+'models/trajectories/'+descriptor.file);
    if (!response.ok) throw new Error('流明动作数据读取失败 '+response.status);
    const bytes = await response.arrayBuffer();
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)))
      .map(value => value.toString(16).padStart(2, '0')).join('');
    if (hash !== descriptor.sha256 || bytes.byteLength !== descriptor.frames*53*16)
      throw new Error('流明动作来源校验失败: '+identity);
    const data = new Float32Array(bytes);
    if (data.some(value => !Number.isFinite(value))) throw new Error('流明动作含非有限数值');
    this.sources.set(identity, data);
    if (this.sources.size > 8) this.sources.delete(this.sources.keys().next().value!);
    return data;
  }
}

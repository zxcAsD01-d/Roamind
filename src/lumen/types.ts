import type { Quaternion, Vector3 } from 'three';

export type LumenSkill = { source: string; start: number; end: number; duration: number;
  score?: number; bias?: { joint: number; angles: number[] } };
export type LumenEntry = { key: string; side?: string; part?: string; mask: number[]; skills: LumenSkill[] };
export type LumenClass = { key: string; canonical: string; mask: number[]; side?: string; file: string };
export type LumenManifest = { version: number; bones: string[]; rate: number; classes: LumenClass[] };
export type LumenSource = { id: string; file: string; sha256: string; frames: number };
export type LayerOptions = { speed?: number; strength?: number; hold?: number; delay?: number; cycles?: number };
export type LayerTrack = { id: number; key: string; mask: number[]; age: number; weight: number;
  fade_seconds: number; source: string; source_frame: number };
export type LayerState = { time: number; tracks: LayerTrack[]; weights: number[] };
export type LayerEngine = { add(entry: LumenEntry, options: LayerOptions, base: number[][]): number;
  sample(base: number[][], dt: number): number[][]; active(): boolean; state(): LayerState; clear(): void; reset(): void };
export type SmoothPoint = { q: Quaternion[]; p: Vector3 };
export type SmoothingEngine = { reset(): void; report(): { recovery_fraction: number } | null;
  step(point: SmoothPoint, dt: number, quality: number[] | null, map: (p: SmoothPoint) => SmoothPoint,
    test: (p: SmoothPoint) => boolean, enabled?: boolean): SmoothPoint };
export type LumenAction = 'wave' | 'raise' | 'point' | `lumen:${string}`;
export function isLumenAction(action: string | undefined): action is LumenAction {
  return action === 'wave' || action === 'raise' || action === 'point' || !!action?.startsWith('lumen:');
}

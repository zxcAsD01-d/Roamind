declare module '*HAST-SEC/motion/beat_layers.mjs' {
  export function beatLayers(bones: string[], sources: Map<string, Float32Array>): import('./types').LayerEngine;
}
declare module '*HAST-SEC/motion/v9/output_smoothing.mjs' {
  export function outputSmoothing(names: string[], settings: { output_smoothing: Record<string, number> }): import('./types').SmoothingEngine;
}
declare module '*HAST-SEC/motion/skill_rotation.mjs' {
  import type { Quaternion } from 'three';
  export function clipRotation(data: Float32Array, frame: number, joint: number, count: number, skill: import('./types').LumenSkill): Quaternion;
}
declare module '*HAST-SEC/motion/v9/damped_output.mjs' {
  import type { Vector3 } from 'three';
  export function dampedStep(error: Vector3, velocity: Vector3, hz: number, dt: number): { move: Vector3; velocity: Vector3 };
}

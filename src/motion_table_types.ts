import type { SmplMotionJson } from './retarget';

export type TableRole = 'idle' | 'start' | 'travel' | 'stop' | 'turn' | 'action';
export type TableEntry = {
  id: string; role: TableRole; name: string; source: string; sourceFrames?: number[]; prompt?: string;
  fps: number; frames: number; duration: number; sourceHeight: number; heightOffset: number;
  offset: number; entry: number[]; exit: number[]; trajectory: number[][]; length: number; yaw: number;
  quality: { supportMean: number; supportP95: number; angularStepMax: number; rootSpeedMax: number };
};
export type TableDocument = {
  version: number; fps: number; up: string; joints: number[]; bones: string[]; parents: number[];
  restJoints: number[]; dataFile: string; dataSha256: string; dataFloats: number;
  entries: TableEntry[]; successors: { to: number; cost: number; poseRms: number }[][];
};
export type TableMotion = SmplMotionJson & { sourceHeight: number; heightOffset: number };
export type TableDiagnostics = {
  entries: string[]; searchMs: number; expanded: number; pathMaxError: number; goalError: number;
  seamPoseRms: number; seamJointMax: number; seamVelocityMax: number;
};

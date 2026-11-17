/** Watch commanded walking using the character's own displacement and simulation time. */
export class MovementProgress {
  private command: string | null = null;
  private x = 0;
  private z = 0;
  private elapsed = 0;

  update(command: string | null, walking: boolean, delta: number, x: number, z: number): boolean {
    if (!command || !walking || command !== this.command) {
      this.command = command; this.x = x; this.z = z; this.elapsed = 0;
      return false;
    }
    if (Math.hypot(x-this.x, z-this.z) >= .12) {
      this.x = x; this.z = z; this.elapsed = 0;
      return false;
    }
    this.elapsed += delta;
    return this.elapsed >= 3;
  }
}

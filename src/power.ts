export type PerceptionCadence = { preview: number; depth: number; semantic: number };

/** Shared policy; active perception keeps the original safety cadence. */
export class PowerMode {
  enabled = true;
  private interactionUntil = 0;

  constructor(private activity: () => boolean) {
    const toggle = document.querySelector<HTMLInputElement>('#power-toggle')!;
    try { this.enabled = localStorage.getItem('motion-power-saving') !== 'off'; }
    catch (error) { this.storageError(error instanceof Error ? error.message : String(error)); }
    toggle.checked = this.enabled;
    toggle.addEventListener('change', () => {
      this.enabled = toggle.checked;
      try { localStorage.setItem('motion-power-saving', this.enabled ? 'on' : 'off'); }
      catch (error) { this.storageError(error instanceof Error ? error.message : String(error)); }
    });
    for (const event of ['pointerdown', 'pointermove', 'wheel', 'keydown']) {
      document.addEventListener(event, () => { this.interactionUntil = performance.now()+1200; }, { passive: true });
    }
  }

  get active(): boolean { return this.activity() || performance.now() < this.interactionUntil; }
  get fps(): number { return this.enabled ? (this.active ? 30 : 15) : 0; }
  get cadence(): PerceptionCadence {
    return this.enabled && !this.active ? { preview: 1000, depth: 1000, semantic: 1500 }
      : { preview: 200, depth: 200, semantic: 500 };
  }

  private storageError(error: string): void {
    console.error('节能偏好保存不可用', error);
    document.querySelector('#power-status')!.textContent = '节能设置仅在本次会话生效（本地存储不可用）';
  }
}

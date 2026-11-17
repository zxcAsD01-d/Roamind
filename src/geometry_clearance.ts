/** Exact planar swept circle against a rectangle, including rounded corners. */
export function segmentBoxDistanceSquared(a: readonly number[], b: readonly number[], center: readonly number[], half: readonly number[]): number {
  let enter = 0, leave = 1;
  for (let axis = 0; axis < 2; axis++) {
    const delta = b[axis]-a[axis], low = center[axis]-half[axis], high = center[axis]+half[axis];
    if (Math.abs(delta) < 1e-12) {
      if (a[axis] < low || a[axis] > high) { enter = Infinity; break; }
    } else {
      const first = (low-a[axis])/delta, last = (high-a[axis])/delta;
      enter = Math.max(enter, Math.min(first, last)); leave = Math.min(leave, Math.max(first, last));
    }
  }
  if (enter <= leave) return 0;
  const pointBox = (p: readonly number[]): number => Math.max(0, Math.abs(p[0]-center[0])-half[0])**2
    + Math.max(0, Math.abs(p[1]-center[1])-half[1])**2;
  let best = Math.min(pointBox(a), pointBox(b));
  const dx = b[0]-a[0], dz = b[1]-a[1], length = dx*dx+dz*dz;
  for (const x of [-1, 1]) for (const z of [-1, 1]) {
    const px = center[0]+x*half[0]-a[0], pz = center[1]+z*half[1]-a[1];
    const t = length ? Math.max(0, Math.min(1, (px*dx+pz*dz)/length)) : 0;
    best = Math.min(best, (px-dx*t)**2+(pz-dz*t)**2);
  }
  return best;
}

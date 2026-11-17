async page => page.evaluate(async () => {
  const { safeStereo } = await import('/src/stereo.ts');
  const now = Date.now()/1000;
  const sample = (clearance, obstacleDistance) => ({ observedAt: now, mapping: true,
    groundInfo: { frame: 'gravity-aligned-y-up', support: 400, tiltDeg: 1, residual: .01 },
    sectors: [{ angle: 0, clearance, confidence: .9, obstacleDistance }] });
  if (!safeStereo(sample(.4, null), now)) throw new Error('未知路线仍必须检查可见地板范围');
  if (safeStereo(sample(.4, null), now, 0, true)) throw new Error('把已知地板的视野边界当成了障碍');
  if (safeStereo(sample(.4, 2), now, 0, true)) throw new Error('测得障碍在远处时仍误停');
  if (!safeStereo(sample(.4, .4), now, 0, true)) throw new Error('已知路线忽略了新障碍');
  if (!safeStereo(sample(null, null), now, 0, true)) throw new Error('空深度被当成安全');
  if (!safeStereo({ ...sample(4, null), mapping: false }, now, 0, true)) throw new Error('定位失败未停车');
  if (!safeStereo(sample(4, null), now+2, 0, true)) throw new Error('过期帧未停车');
  if (!safeStereo(sample(4, null), now, 60, true)) throw new Error('行进方向在视野外未停车');
  return { passed: true, checks: 8 };
})

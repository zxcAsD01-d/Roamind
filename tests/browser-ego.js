async (page) => {
  await page.locator('details.research-controls').evaluateAll(elements => elements.forEach(element => { element.open = true; }));
  await page.locator('#ambient-idle').uncheck();
  const pixels = () => page.locator('#ego-preview').evaluate(canvas => {
    const data = canvas.getContext('2d').getImageData(0, 0, 640, 360).data;
    return Array.from(data).filter((_, index) => index % 64 < 3);
  });
  const rotate = async (axis, value) => {
    await page.locator(`#axis-controls input[data-axis="${axis}"]`).evaluate((input, degrees) => {
      input.value = String(degrees); input.dispatchEvent(new Event('input', { bubbles: true }));
    }, value);
    await page.waitForTimeout(1200);
  };
  await page.locator('#reset-house').click();
  await page.locator('#ego-toggle').check();
  await page.waitForFunction(() => document.querySelector('#vision-status').textContent.includes('个物体'));
  await page.locator('#joint-select').selectOption('head');
  const original = {};
  for (const axis of ['x', 'y', 'z']) original[axis] = Number(await page.locator(`#axis-controls input[data-axis="${axis}"]`).inputValue());
  const changes = {};
  try {
    for (const axis of ['x', 'y', 'z']) {
      const before = await pixels();
      await rotate(axis, original[axis] + 25);
      const after = await pixels();
      changes[axis] = after.reduce((sum, value, index) => sum + Math.abs(value - before[index]), 0) / after.length;
      if (changes[axis] < 1) throw new Error(`头部 ${axis} 旋转没有改变第一人称画面`);
      await rotate(axis, original[axis]);
    }
    return { headRotationPixelDifference: changes, microphoneClosed: await page.locator('#mic-stop').isDisabled() };
  } finally {
    for (const axis of ['x', 'y', 'z']) await rotate(axis, original[axis]);
  }
}

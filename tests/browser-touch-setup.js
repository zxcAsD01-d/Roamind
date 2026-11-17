async (page) => {
  await page.goto('http://127.0.0.1:5173/');
  await page.waitForFunction(() => document.querySelector('#load-status')?.textContent.includes('步幅比例'));
  await page.locator('#reset-house').click();
  await page.locator('#touch-partner').click();
  await page.locator('#power-toggle').check();
  await page.locator('#ego-toggle').check();
  await page.waitForFunction(() => [...document.querySelector('#touch-person').options].some(option => option.value), null, {timeout:20000});
  return {clientId:await page.locator('#viewport').getAttribute('data-client-id')};
}

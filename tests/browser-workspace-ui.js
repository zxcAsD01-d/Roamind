async page => {
  const result = {tabs: false, sizes: [], duplicateIds: [], controls: {}};
  await page.locator('details').evaluateAll(items => items.forEach(item => { item.open = false; }));
  await page.locator('.left-panel').evaluate(node => {node.scrollTop=0;});
  await page.getByRole('tab', { name: '世界记忆' }).click();
  await page.getByRole('tab', { name: '世界记忆' }).press('ArrowRight');
  if (await page.getByRole('tab', {name:'行为决策'}).getAttribute('aria-selected') !== 'true') throw new Error('方向键没有切换到行为页签');
  await page.getByRole('tab', { name: '行为决策' }).press('End');
  if (await page.getByRole('tab', {name:'动作工具'}).getAttribute('aria-selected') !== 'true') throw new Error('End 键没有切换到动作页签');
  if (!await page.locator('#action-library-panel').isVisible()) throw new Error('动作库没有挂载到动作工具页');
  result.tabs = true;
  await page.getByRole('tab', { name: '世界记忆' }).click();
  await page.setViewportSize({width:1440,height:1000});
  await page.waitForTimeout(700);
  await page.screenshot({path:'output/playwright/hierarchical-rebuild/desktop-world.png'});
  await page.getByRole('tab', { name: '行为决策' }).click();
  await page.screenshot({path:'output/playwright/hierarchical-rebuild/desktop-social.png'});
  await page.getByRole('tab', { name: '世界记忆' }).click();
  for (const [width,height] of [[900,1000],[390,844],[320,740]]) {
    await page.setViewportSize({width,height});
    await page.waitForTimeout(500);
    const size = await page.evaluate(() => ({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,
      stageHeight:document.querySelector('.stage').getBoundingClientRect().height}));
    if (size.scrollWidth > width+1) throw new Error('页面出现横向溢出：'+JSON.stringify(size));
    result.sizes.push(size);
    await page.screenshot({path:`output/playwright/hierarchical-rebuild/responsive-${width}.png`,fullPage:width===390});
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.getByRole('tab', {name:'行为决策'}).click();
  result.duplicateIds = await page.evaluate(() => [...document.querySelectorAll('[id]')].map(node=>node.id).filter((id,index,ids)=>ids.indexOf(id)!==index));
  if (result.duplicateIds.length) throw new Error('重复组件 ID：'+result.duplicateIds.join(','));
  result.controls = await page.evaluate(() => ({model:document.querySelector('#stage-model').textContent,
    powerSaving:document.querySelector('#power-toggle').checked,
    microphone:!document.querySelector('#mic-stop').disabled,
    vision:document.querySelector('#ego-toggle').checked,
    connection:document.querySelector('#workspace-connection').textContent}));
  return result;
}

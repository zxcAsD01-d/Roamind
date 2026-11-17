async page=>{
 await page.goto('http://127.0.0.1:5173/');
 await page.waitForFunction(()=>document.querySelector('#load-status')?.textContent.includes('步幅比例'),null,{timeout:60000});
 await page.evaluate(()=>{const input=document.querySelector('#ego-toggle');if(input.checked)input.click();});
 return await page.evaluate(async()=>{
 const THREE=await import('/node_modules/.vite/deps/three.js');
 const {GLTFLoader}=await import('/node_modules/.vite/deps/three_addons_loaders_GLTFLoader__js.js');
 const {VRMLoaderPlugin}=await import('/node_modules/.vite/deps/@pixiv_three-vrm.js');
 const {SmplClip,VrmRetargeter,measureVrmRest}=await import('/src/retarget.ts');
 const {LocomotionLibrary}=await import('/src/locomotion.ts');
 const {InteractionBody}=await import('/src/interaction_body.ts');
 const {HouseScene}=await import('/src/house.ts');
 const loader=new GLTFLoader();loader.register(parser=>new VRMLoaderPlugin(parser));
 const vrm=(await loader.loadAsync('/models/Wolferia.vrm')).userData.vrm;
 const scene=new THREE.Scene(),root=new THREE.Group();scene.add(root);root.add(vrm.scene);vrm.scene.rotation.y=Math.PI;
 const bounds=new THREE.Box3().setFromObject(vrm.scene),center=bounds.getCenter(new THREE.Vector3());
 vrm.scene.position.set(-center.x,-bounds.min.y,-center.z);measureVrmRest(vrm);
 const library=await LocomotionLibrary.load(vrm),world=await(await fetch('http://127.0.0.1:8100/api/world')).json();
 // Scene truth is exclusively test placement/verification, never passed as an observation.
 world.props=[{id:'remote-control',label:'remote control',center:[-6.08,.468,5.1],size:[.055,.036,.18]}];
 const house=new HouseScene(world,scene);
 const body=new InteractionBody({vrm:()=>vrm,root,obstacles:()=>world.objects,props:()=>house.props});
 const place=(position,heading)=>{body.reset();const motion=library.idle(new THREE.Vector3(...position),heading);const driver=new VrmRetargeter(vrm,SmplClip.from(motion.motion));root.position.copy(driver.rootAt(0));driver.apply(0);vrm.update(0);root.updateWorldMatrix(true,true);};
 const camera=new THREE.PerspectiveCamera(42,1.2,.01,50),renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(840,700);renderer.domElement.id='house-node-render';renderer.domElement.style.cssText='position:fixed;left:0;top:0;z-index:99999';document.body.append(renderer.domElement);
 scene.background=new THREE.Color(0x1e2a30);scene.add(new THREE.HemisphereLight(0xd8e9ff,0x77725d,2));const light=new THREE.DirectionalLight(0xffe9ca,2.5);light.position.set(-4,6,3);scene.add(light);
 const images={};
 const run=async(op,node)=>{let done=false,error;const p=body.execute('house-'+op,op,'test-fixture',node).then(()=>{done=true},e=>{done=true;error=e.message});for(let i=0;i<800&&!done;i++){body.update(1/60);vrm.update(1/60);await Promise.resolve();}if(!done)throw new Error('body did not terminate');await p;renderer.render(scene,camera);images[op]=renderer.domElement.toDataURL();return {error:error??null,state:body.snapshot()};};
 place([-8.02,0,4.7],Math.PI/2);camera.position.set(-5.7,2.8,7.5);camera.lookAt(-8.5,.5,4.7);
 const node={kind:'support',controlPoint:[-8.704,.48025,4.7],controlAxis:[0,0,1],size:[2.6,0,.82],sigma:.005,observedAt:Date.now()/1000,source:'measured-stereo-surface',frame:'control-world-y-up'};
 const result={sit:await run('sit',node)};result.stand=await run('stand');result.lie=await run('lie',{...node,observedAt:Date.now()/1000});result.standAfterLie=await run('stand');
 place([-5.72,0,5.1],-Math.PI/2);camera.position.set(-7.6,1.8,4.0);camera.lookAt(-5.85,.65,5.1);
 result.pickup=await run('pickup',{...node,kind:'grasp',controlPoint:[-6.08,.486,5.1],size:[.055,.036,.18],observedAt:Date.now()/1000});result.release=await run('release');
 for(const [name,row] of Object.entries(result)){if(row.error)throw new Error(name+': '+row.error);}
 window.houseNodeTest={scene,camera,renderer,images,body,vrm,root,result};return result;
 });
}

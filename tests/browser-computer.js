async page => {
 await page.goto('http://127.0.0.1:5173/');
 await page.waitForFunction(()=>document.querySelector('#load-status')?.textContent.includes('步幅比例'),null,{timeout:60000});
 await page.evaluate(()=>{const input=document.querySelector('#ego-toggle');if(input.checked)input.click();});
 return page.evaluate(async()=>{
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
 // Explicit fixture placement: scene truth is never submitted as a visual observation.
 const house=new HouseScene(world,scene);
 const body=new InteractionBody({vrm:()=>vrm,root,obstacles:()=>world.objects,props:()=>house.props,computers:()=>house.computers});
 const chair=world.objects.find(o=>o.id==='computer-chair:cushion');
 const seatPoint=[chair.center[0],chair.center[1]+chair.size[1]/2,chair.center[2]];
 const motion=library.idle(new THREE.Vector3(8,0,seatPoint[2]-.46),Math.PI);
 const driver=new VrmRetargeter(vrm,SmplClip.from(motion.motion));root.position.copy(driver.rootAt(0));driver.apply(0);vrm.update(0);root.updateWorldMatrix(true,true);
 const camera=new THREE.PerspectiveCamera(42,1.5,.01,50),renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(1000,700);
 renderer.domElement.id='computer-test-render';renderer.domElement.style.cssText='position:fixed;left:0;top:0;z-index:99999';document.body.append(renderer.domElement);
 scene.background=new THREE.Color(0x1e2a30);scene.add(new THREE.HemisphereLight(0xd8e9ff,0x77725d,2));const light=new THREE.DirectionalLight(0xffe9ca,2.5);light.position.set(6,6,-4);scene.add(light);
 house.setCutaway(true);camera.position.set(6.2,2.2,-4.9);camera.lookAt(8,.8,-6.6);
 const seat={kind:'support',controlPoint:seatPoint,controlAxis:[1,0,0],size:[chair.size[0],0,chair.size[2]],sigma:.005,observedAt:Date.now()/1000,source:'measured-stereo-surface',frame:'control-world-y-up'};
 const node={...seat,kind:'workstation',controlPoint:world.computers[0].keyboard,size:[.42,0,.16],seat};
 let done=false,error,peak=null;
 const task=body.execute('fixture-play','play_computer','fixture-keyboard',node,12).then(()=>done=true,e=>{done=true;error=e.message});
 for(let i=0;i<2200&&!done;i++) {body.update(1/60);vrm.update(1/60);await Promise.resolve();if(body.snapshot().computer?.active){peak=body.snapshot();renderer.render(scene,camera);}}
 if(!done)throw new Error('computer execution timeout');await task;renderer.render(scene,camera);
 if(error)throw new Error(error);const final=body.snapshot();if(!final.supported||!final.computer.completed||final.computer.elapsed<12||final.computer.inputs<1||final.computer.score<1)throw new Error('game result missing');
 window.computerFixture={body,vrm,root,scene,camera,renderer,house,node};
 const {point}=await import("/src/interaction_pose.ts"); const bones=Object.fromEntries(["hips","spine","chest","rightUpperArm","rightLowerArm","rightHand","leftUpperArm"].map(n=>[n,point(vrm,n).toArray()]));
 return {bones,error:error??null,state:body.snapshot(),during:peak,seat:seatPoint,root:root.position.toArray()};
 });
}

import * as THREE from 'three';
import {createCharacter} from './avatarScene';
import {applyDesign} from './avatarDesigns';
import {ITEM_TYPES} from './avatarItemTypes';
import {DEFAULT_CHARACTER} from './avatarOptions';
jest.mock('three/addons/utils/BufferGeometryUtils.js',()=>({mergeVertices:geometry=>geometry.clone()}),{virtual:true});
for(const[kind,type]of Object.entries(ITEM_TYPES))test(`every ${kind} fit attaches a textured, bounded 3D mesh`,()=>{
 for(const[label,region,baseStyle]of type.fits){
  const design={kind,region,baseStyle,name:label,texture:'data:image/png;base64,YWJj',length:.5,fringe:.5,textureAmount:.5},look=applyDesign(DEFAULT_CHARACTER,design),slot=Object.keys(look).find(key=>look[key]===design),texture=new THREE.Texture(),model=createCharacter(look,{[slot]:texture});
  try{let mapped=0;model.traverse(mesh=>{if(mesh.material?.map===texture){mapped++;expect(mesh.geometry.attributes.uv).toBeDefined();}});expect(mapped).toBeGreaterThan(0);const size=new THREE.Box3().setFromObject(model).getSize(new THREE.Vector3());expect(size.x).toBeLessThan(3);expect(size.y).toBeLessThan(4);expect(size.z).toBeLessThan(3);expect(size.y).toBeGreaterThan(2);}finally{model.userData.dispose();texture.dispose();}
 }
});

test('a centred shirt sketch is centred on the visible front, without mutating shared geometry',()=>{
 const texture=new THREE.Texture(),look=applyDesign(DEFAULT_CHARACTER,{kind:'clothing',region:'top',baseStyle:0,name:'Front star',texture:'data:image/png;base64,YWJj'}),model=createCharacter(look,{customClothing:texture});
 try{const body=model.children.find(mesh=>mesh.material?.map===texture&&mesh.geometry.type==='CapsuleGeometry'&&mesh.position.x===0),p=body.geometry.attributes.position,uv=body.geometry.attributes.uv;let checked=0;for(let i=0;i<p.count;i++)if(p.getZ(i)>.45&&Math.abs(p.getX(i))<.01){expect(Math.abs(uv.getX(i)-.5)).toBeLessThan(.03);checked++;}expect(checked).toBeGreaterThan(0);}finally{model.userData.dispose();texture.dispose();}
});

import * as THREE from 'three';
import {createCharacter} from './avatarScene';
import {AVATAR_OPTIONS,DEFAULT_CHARACTER,randomCharacter} from './avatarOptions';
jest.mock('three/addons/utils/BufferGeometryUtils.js',()=>({mergeVertices:geometry=>geometry.clone()}),{virtual:true});
function signature(model){let hash=2166136261;model.traverse(node=>{const p=node.geometry?.attributes.position;if(!p)return;for(const value of [...node.position,...node.scale,node.material.color.getHex(),p.count])hash=Math.imul(hash^Math.round(value*1000),16777619);for(let i=0;i<p.count;i+=Math.max(1,Math.floor(p.count/41)))for(const value of [p.getX(i),p.getY(i),p.getZ(i)])hash=Math.imul(hash^Math.round(value*10000),16777619);});return hash>>>0;}
function inspect(look,check){const model=createCharacter({...DEFAULT_CHARACTER,...look});try{const bounds=new THREE.Box3().setFromObject(model);expect([...bounds.min,...bounds.max].every(Number.isFinite)).toBe(true);expect(bounds.max.y).toBeLessThan(4);expect(bounds.min.y).toBeGreaterThan(-.1);expect(bounds.max.x-bounds.min.x).toBeLessThan(3);check(model);}finally{model.userData.dispose();}}
test('all tops have a shaped fabric body and distinct outfits, including the complete expanded wardrobe',()=>{
 const seen=new Set();for(let topStyle=0;topStyle<AVATAR_OPTIONS.topStyle.length;topStyle++)inspect({topStyle},model=>{const body=model.children.find(mesh=>mesh.userData.avatarPart==='shirt-body');expect(body).toBeDefined();expect(body.geometry.type).toBe('BufferGeometry');expect(body.geometry.attributes.uv.count).toBe(body.geometry.attributes.position.count);const current=signature(model);expect(seen.has(current)).toBe(false);seen.add(current);});
});
test('every hair and newly added wearable changes actual geometry within the character bounds',()=>{
 for(const key of ['hairStyle','hat','glasses','shoeStyle','pantsStyle','accessoryStyle','necklace','earring','eyeShape','eyebrow','mouth','facialHair']){const seen=new Set();for(let i=key==='pantsStyle'?1:0;i<AVATAR_OPTIONS[key].length;i++)inspect({[key]:i},model=>{const current=signature(model);if(seen.has(current))throw new Error('Duplicate geometry for '+key+': '+AVATAR_OPTIONS[key][i]);seen.add(current);});}
});
test('existing option indices remain compatible and the randomizer reaches expanded clothes and shoes',()=>{
 expect(AVATAR_OPTIONS.hairStyle[13]).toBe('Two buns');expect(AVATAR_OPTIONS.hairStyle[16]).toBe('Swept back');expect(AVATAR_OPTIONS.topStyle[15]).toBe('Denim');
 const look=randomCharacter({...DEFAULT_CHARACTER,name:'Saved',designs:[{id:'keep'}]},()=>.9999);expect(look.topStyle).toBe(35);expect(look.hat).toBe(17);expect(look.shoeStyle).toBe(9);expect(look.name).toBe('Saved');expect(look.designs).toEqual([{id:'keep'}]);
});

import * as THREE from "three";
import {addAvatarDetails} from "./avatarDetails";
import {addWardrobe} from "./avatarWardrobe";
import { mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";

// Original, procedural Classio characters. No third-party character assets.
const sphere = new THREE.SphereGeometry(1, 36, 24);
const capsule = new THREE.CapsuleGeometry(.5, 1, 8, 20);
const box = new THREE.BoxGeometry(1, 1, 1);
const hairCap = new THREE.SphereGeometry(1, 44, 24, 0, Math.PI * 2, 0, 1.30);
// Keep the hairline higher at the forehead and lower around the sides.
for (let i = 0; i < hairCap.attributes.position.count; i++) {
  const p = hairCap.attributes.position, x = p.getX(i), y = p.getY(i), z = p.getZ(i), theta = Math.acos(Math.min(1, Math.max(-1, y)));
  const radial = Math.hypot(x, z); if (radial < .0001) continue;
  const front = Math.max(0, z / radial), next = theta / 1.30 * (1.36 - .26 * front ** 4), scale = Math.sin(next) / radial;
  p.setXYZ(i, x * scale, Math.cos(next), z * scale);
}
hairCap.computeVertexNormals();
const validColor = (value, fallback) => /^#[\da-f]{3}(?:[\da-f]{3})?$/i.test(value || "") ? value : fallback;
const makeMaterial = (color, roughness = .78) => new THREE.MeshStandardMaterial({ color, roughness, metalness: 0 });

export function createCharacter(character = {}, textures = {}) {
  const ch = character, root = new THREE.Group(), owned = new Set(), createdMaterials=new Set();
  const material=(...args)=>{const value=makeMaterial(...args);createdMaterials.add(value);return value;};
  const skinColor = validColor(ch.skin, "#edc5a1"), hairColor = validColor(ch.hair, "#493326");
  const skin = material(skinColor), brows = material(hairColor,.8), hair = material(hairColor, .72), shirt = material(validColor(ch.top, "#528ca6"));
  const white = material("#fffaf1", .45), ink = material("#242c34", .45), iris = material(validColor(ch.eyes, "#487b9b"), .38);
  const pants = material(validColor(ch.bottom,"#3b4b60")), shoes = material(validColor(ch.shoes,"#e3d4b7")), lips = material(ch.lips ? validColor(ch.lipColor, "#ae6262") : "#754f44");
  const style = Number(ch.hairStyle) || 0;
  function fitted(mat, field) { if (textures[field]) { mat.map = textures[field]; mat.color.set('#ffffff'); mat.needsUpdate = true; } return mat; }
  fitted(shirt,'customClothing'); fitted(hair,'customHair'); fitted(pants,'customBottom'); fitted(shoes,'customShoes');
  if (ch.customHair) hair.roughness = .4 + Math.min(1,Math.max(0,ch.customHair.textureAmount || 0)) * .55;
  function mesh(geometry, mat, position, scale = [1, 1, 1], parent = root, own = false) {
    // A front sketch must face the viewer; lathe UVs put its centre on the back.
    // Clone shared primitives so built-in characters keep their original UVs.
    if(mat.map && ["CapsuleGeometry","SphereGeometry","CylinderGeometry","ConeGeometry","BoxGeometry"].includes(geometry.type)) {
      if(own)owned.add(geometry);geometry=geometry.clone();owned.add(geometry);geometry.computeBoundingBox();
      const bounds=geometry.boundingBox,p=geometry.attributes.position,uv=[];
      for(let i=0;i<p.count;i++)uv.push((p.getX(i)-bounds.min.x)/Math.max(.0001,bounds.max.x-bounds.min.x),(p.getY(i)-bounds.min.y)/Math.max(.0001,bounds.max.y-bounds.min.y));
      geometry.setAttribute("uv",new THREE.Float32BufferAttribute(uv,2));
    }
    const item = new THREE.Mesh(geometry, mat); item.position.set(...position); item.scale.set(...scale); item.castShadow = true; item.receiveShadow = true; parent.add(item); if (own) owned.add(geometry); return item;
  }
  const oval = (mat, x, y, z, rx, ry, rz, parent) => mesh(sphere, mat, [x, y, z], [rx, ry, rz], parent);
  const limb = (mat, x, y, z, rx, height, rz) => mesh(capsule, mat, [x, y, z], [rx * 2, height / 2, rz * 2]);
  function tube(mat, points, radius = .018, parent = root, closed = false) {
    const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)), closed);
    return mesh(new THREE.TubeGeometry(curve, Math.max(20, points.length * 5), radius, 8, closed), mat, [0, 0, 0], [1, 1, 1], parent, true);
  }
  function ring(mat, x, y, z, rx, ry, thickness = .018, parent = root, shape = "round") {
    const points = Array.from({ length: 40 }, (_, i) => { const a = i / 40 * Math.PI * 2; if(shape === "star"){const r=i%8<4?1:.55;return[x+Math.cos(a)*rx*r,y+Math.sin(a)*ry*r,z];} if(shape === "heart") return [x + rx * Math.sin(a) ** 3,y + ry * (13*Math.cos(a)-5*Math.cos(2*a)-2*Math.cos(3*a)-Math.cos(4*a)+2)/17,z]; return [x + rx * Math.sign(Math.cos(a)) * Math.pow(Math.abs(Math.cos(a)), shape === "rectangle" ? .45 : 1), y + ry * Math.sign(Math.sin(a)) * Math.pow(Math.abs(Math.sin(a)), shape === "rectangle" ? .45 : 1) + (shape === "cat-eye" ? .35*ry*Math.abs(Math.cos(a))**4 : shape === "aviator" && Math.sin(a)<0 ? -.18*ry*Math.abs(Math.sin(a)) : 0), z]; });
    return tube(mat, points, thickness, parent, true);
  }

  addWardrobe({ch,root,mesh,oval,limb,tube,ring,material,skin,shirt,pants,shoes,white,ink});

  const solidHat=Number(ch.hat)>0&&![7,13,14,16].includes(Number(ch.hat));
  const long = [4,5,9,11,15,17,20,21,26,27,28,31].includes(style);
  if (long) oval(hair, 0, 2.00, -.18, .655, ch.customHair ? .64 + Math.min(1,Math.max(0,ch.customHair.length || 0)) * .35 : [17,31].includes(style) ? .61 : .84, .43);
  oval(skin, 0, 2.16, 0, .64, .60, .55);
  for (const sign of [-1, 1]) {
    oval(skin, sign * .635, 2.12, -.015, .14, .17, .12);
    const inner = material(new THREE.Color(skinColor).multiplyScalar(.79)); oval(inner, sign * .67, 2.12, .069, .065, .085, .025);
  }
  if(style!==32)mesh(hairCap, hair, [0, 2.16, -.008], [.67, style===29?.602:.635, .575]);
  // The fringe and every extension intersect the crown, rather than floating.
  if (![0,8,14,16,24,25,29,32].includes(style)) {
    const fringe = new THREE.Shape();
    fringe.moveTo(-.60, 2.44); fringe.bezierCurveTo(-.52, 2.68, -.16, 2.80, .20, 2.72); fringe.bezierCurveTo(.44, 2.66, .59, 2.53, .59, 2.40);
    if ([2,17,31].includes(style)) { fringe.lineTo(.40, 2.32); fringe.quadraticCurveTo(.31, 2.28, .24, 2.22); fringe.lineTo(.12, 2.34); fringe.quadraticCurveTo(-.05, 2.27, -.19, 2.25); fringe.lineTo(-.27, 2.36); fringe.lineTo(-.48, 2.29); }
    else if ([4,19,21,27].includes(style)) { fringe.quadraticCurveTo(.48, 2.22, .35, 2.23); fringe.quadraticCurveTo(.17, 2.46, 0, 2.55); fringe.quadraticCurveTo(-.17, 2.46, -.36, 2.24); fringe.lineTo(-.55, 2.29); }
    else if ([5,15,18,20,28].includes(style)) { fringe.quadraticCurveTo(.36, 2.32, .20, 2.25); fringe.quadraticCurveTo(.01, 2.28, -.05, 2.42); fringe.quadraticCurveTo(-.28, 2.49, -.57, 2.37); }
    else if (style === 3 || style === 10) { fringe.quadraticCurveTo(.36, 2.48, .30, 2.36); fringe.quadraticCurveTo(.10, 2.48, .02, 2.36); fringe.quadraticCurveTo(-.14, 2.47, -.25, 2.33); fringe.lineTo(-.53, 2.40); }
    else if (style === 12) { fringe.lineTo(.36, 2.46); fringe.lineTo(.20, 2.42); fringe.lineTo(.04, 2.49); fringe.lineTo(-.16, 2.43); fringe.lineTo(-.48, 2.46); }
    else { fringe.quadraticCurveTo(.38, 2.46, .20, 2.55); fringe.quadraticCurveTo(-.08, 2.44, -.30, 2.46); fringe.lineTo(-.58, 2.40); }
    fringe.closePath();
    const flat = new THREE.ExtrudeGeometry(fringe, { depth: .045, bevelEnabled: true, bevelSegments: 4, steps: 1, bevelSize: .022, bevelThickness: .018, curveSegments: 24 });
    // Tessellate before wrapping: large flat triangles otherwise cut across the
    // curved forehead and make the fringe look faceted or intersect the face.
    const vertices = [], source = flat.attributes.position;
    const midpoint = (a, b) => a.map((v, i) => (v + b[i]) / 2);
    function subdivide(a, b, c, depth = 0) {
      const distance = (u, v) => Math.hypot(...u.map((n, i) => n - v[i]));
      if (depth < 4 && Math.max(distance(a, b), distance(b, c), distance(c, a)) > .10) {
        const ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
        subdivide(a, ab, ca, depth + 1); subdivide(ab, b, bc, depth + 1);
        subdivide(ca, bc, c, depth + 1); subdivide(ab, bc, ca, depth + 1);
      } else vertices.push(...a, ...b, ...c);
    }
    for (let i = 0; i < source.count; i += 3) subdivide(...[0, 1, 2].map(j => [source.getX(i + j), source.getY(i + j), source.getZ(i + j)]));
    const wrapped = new THREE.BufferGeometry(); wrapped.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    const geometry = mergeVertices(wrapped); flat.dispose(); wrapped.dispose();
    const positions = geometry.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i), originalY = positions.getY(i), y = ch.customHair ? 2.77 - (2.77 - originalY) * (.82 + Math.min(1,Math.max(0,ch.customHair.fringe || 0)) * .36) : originalY, surface = .575 * Math.sqrt(Math.max(.025, 1 - (x / .69) ** 2 - ((y - 2.16) / .65) ** 2));
      positions.setY(i,y);
      positions.setZ(i, surface + .035 + positions.getZ(i));
    }
    if(textures.customHair) { const uv=[];for(let i=0;i<positions.count;i++)uv.push((positions.getX(i)+.65)/1.3,(positions.getY(i)-2.12)/.7);geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2)); }
    geometry.computeVertexNormals();
    mesh(geometry, hair, [0, 0, 0], [1, 1, 1], root, true);
  }
  if (style === 0 || style === 16) tube(hair, [[-.53, 2.44, .34], [-.2, 2.52, .5], [.22, 2.55, .49], [.52, 2.43, .31]], style === 0 ? .07 : .10);
  if (!solidHat && (style === 3 || style === 10 || style === 8)) {
    const n = style === 8 ? 13 : style===10?10:7;
    for (let i = 0; i < n; i++) { const a = i / (n - 1) * Math.PI; oval(hair, Math.cos(a) * .51, 2.41 + Math.sin(a) * .30, .19 + Math.sin(a) * .15, style === 8 ? .22 : .15, .19, .19); }
  }
  if (style === 6) tube(hair, [[.22, 2.66, -.39], [.52, 2.55, -.54], [.59, 2.06, -.49], [.43, 1.61, -.38]], .17);
  if (style === 7 && !solidHat) oval(hair, .03, 2.79, -.22, .25, .25, .23);
  if (style === 13 && !solidHat) for (const sign of [-1, 1]) oval(hair, sign * .51, 2.61, -.06, .24, .25, .23);
  if ([9, 11, 15].includes(style)) {
    for (const sign of [-1, 1]) for (let j = 0; j < (style === 9 ? 1 : 3); j++) {
      const x = sign * (.49 + j * .055), z = .02 - j * .13;
      tube(hair, [[x, 2.48, z], [x + sign * .035, 2.0, z + .06], [x, 1.43 + j * .10, z + .035]], .065);
      for (let k = 0; k < 6; k++) oval(hair, x + Math.sin(k * 2.2) * .035, 2.13 - k * .12, z + .05, .075, .09, .075);
    }
  }
  if (style === 5) for (const sign of [-1, 1]) tube(hair, [[sign * .55, 2.4, .03], [sign * .65, 2.13, .14], [sign * .56, 1.92, .11], [sign * .65, 1.65, .07]], .11);
  if ([12, 14].includes(style) && !solidHat) for (let i = 0; i < 4; i++) { const tuft = mesh(new THREE.ConeGeometry(.13, .25, 5), hair, [0, 2.84, -.33 + i * .2], [1, 1, 1], root, true); tuft.rotation.z = -.25; }

  // New silhouettes share the crown but have their own connected lengths and parts.
  if([17,21,31].includes(style))for(const sign of [-1,1]){
    const bottom=style===21?1.51:1.83;
    tube(hair,[[sign*.54,2.43,.04],[sign*.62,2.17,.09],[sign*.60,bottom+.12,.08],[sign*.51,bottom,.10]],style===31?.145:.11);
  }
  if(style===18)tube(hair,[[-.46,2.53,.31],[-.23,2.60,.48],[.08,2.52,.57],[.35,2.37,.51]],.06);
  if([19,20,28].includes(style))for(const sign of [-1,1]){
    tube(hair,[[sign*.16,2.67,.32],[sign*.39,2.50,.48],[sign*.51,2.25,.37],[sign*.55,2.02,.17]],.08);
    if(style!==19)for(let layer=0;layer<(style===28?4:3);layer++)tube(hair,[[sign*.53,2.25-layer*.16,-.08],[sign*.67,2.10-layer*.16,-.01],[sign*.58,1.93-layer*.16,.03]],.08);
  }
  if(style===22){tube(hair,[[0,2.35,-.48],[.15,2.20,-.64],[.22,1.90,-.66],[.09,1.52,-.49]],.16);oval(material('#b66957'),.06,2.23,-.61,.09,.055,.10);}
  if(style===23)for(const sign of [-1,1]){tube(hair,[[sign*.50,2.39,-.15],[sign*.73,2.23,-.26],[sign*.73,1.92,-.16],[sign*.60,1.57,-.10]],.12);oval(material('#b66957'),sign*.65,2.30,-.23,.075,.055,.075);}
  if(style===24&&!solidHat){oval(hair,0,2.91,-.17,.34,.30,.31);for(let i=0;i<9;i++){const a=i/9*Math.PI*2;oval(hair,Math.cos(a)*.25,2.96+Math.sin(a)*.17,-.03,.14,.13,.14);}}
  if(style===25)for(let row=-2;row<=2;row++){
    const x=row*.20,r=Math.sqrt(1-(x/.67)**2);
    tube(hair,Array.from({length:17},(_,i)=>{const a=-1.30+i/16*2.27;return[x,2.16+Math.cos(a)*r*.66,Math.sin(a)*r*.59];}),.037);
  }
  if([26,30].includes(style))for(const sign of style===26?[1]:[-1,1]){
    const x=sign*.55;
    for(let strand=0;strand<2;strand++)tube(hair,Array.from({length:16},(_,i)=>[x+Math.sin(i*1.4+strand*Math.PI)*.035,2.35-i*.058,.10+Math.cos(i*1.4+strand*Math.PI)*.035]),.043);
    oval(material('#b66957'),x,1.49,.10,.055,.025,.055);
    if(style===30&&!solidHat)oval(hair,sign*.48,2.67,-.14,.23,.22,.22);
  }
  if(style===27&&!solidHat)oval(hair,0,2.78,-.30,.23,.22,.23);

  const faceZ = (x, y) => Math.sqrt(Math.max(.05, 1 - (x / .64) ** 2 - ((y - 2.16) / .6) ** 2)) * .55;
  const eyeShape = Number(ch.eyeShape) || 0;
  for (const sign of [-1, 1]) {
    const x = sign * .245, y = 2.19, eye = new THREE.Group(); eye.position.set(x, y, faceZ(x, y) + .016); eye.rotation.y = sign * .23; root.add(eye);
    if (eyeShape === 4) tube(ink, [[-.11, -.02, .035], [0, .055, .05], [.11, -.02, .035]], .025, eye);
    else {
      const ry = eyeShape === 1 ? .137 : eyeShape===6?.12:.165;
      oval(white, 0, 0, 0, eyeShape === 5 ? .142 : eyeShape===6?.15:.13, ry, .035, eye);
      oval(iris, .012, -.006, .034, .082, ry * .78, .018, eye); oval(ink, .014, -.009, .052, .046, ry * .60, .011, eye);
      oval(white, -.012, .046, .064, .025, .032, .008, eye); oval(white, .044, -.06, .066, .011, .014, .006, eye);
      if (eyeShape === 3 || eyeShape===7) oval(skin, 0, eyeShape===7?.085:.127, .049, .14, eyeShape===7?.085:.055, .025, eye);
      if (eyeShape === 2) for (let i = 0; i < 3; i++) tube(ink, [[sign * .10, .09 - i * .035, .018], [sign * (.16 + i * .008), .12 - i * .03, .022]], .012, eye);
    }
    const brow = Number(ch.eyebrow) || 0, raised = brow === 4 && sign === -1 ? .025 : 0, middle = (brow === 1 ? 2.382 : brow===6 ? 2.445 : [3,7].includes(brow) ? 2.425 : brow===8 ? 2.39 : brow===5 ? 2.394 : 2.405) + raised;
    const browPoints = [[x - .10, 2.375 + raised], [x, middle], [x + .10, 2.378 + raised]].map(([bx, by]) => [bx, by, faceZ(bx, by) + .022]);
    tube(brows, browPoints, [2,7].includes(brow) ? .023 : [5,8].includes(brow) ? .010 : .014);
  }
  oval(skin, 0, 2.065, .566, .062, .070, .065);
  if (ch.blush) { const blush = material("#d98b85"); blush.transparent = true; blush.opacity = .35; for (const sign of [-1, 1]) oval(blush, sign * .405, 1.995, faceZ(sign * .405, 1.995) + .008, .095, .045, .012); }
  if (ch.freckles) { const freckles = material("#a57759"); for (const sign of [-1, 1]) for (let i = 0; i < 3; i++) oval(freckles, sign * (.31 + i * .055), 2.00 + (i % 2) * .035, faceZ(sign * (.31 + i * .055), 2.0) + .017, .009, .009, .006); }
  const mouth = Number(ch.mouth) || 0;
  if (mouth === 2) oval(lips, 0, 1.925, .513, .067, .082, .022);
  else if(mouth===7){oval(lips,0,1.94,.516,.13,.055,.022);oval(white,0,1.95,.538,.108,.034,.009);}
  else if(mouth===8)tube(lips,[[-.075,1.95,.529],[0,1.925,.54],[.075,1.95,.529]],.012);
  else if(mouth===9)tube(lips,[[-.055,1.925,.53],[0,1.945,.544],[.055,1.925,.53]],.016);
  else if ([5,6].includes(mouth)) { oval(lips, 0, 1.94, .516, .14, mouth === 5 ? .075 : .11, .025); oval(white, 0, 1.978, .54, .103, .027, .01); if (mouth === 6) oval(material("#cf8f8c"), 0, 1.88, .536, .065, .022, .01); }
  else tube(lips, [[-.13, mouth === 1 ? 1.92 : 1.965, .515], [0, mouth === 1 ? 1.97 : mouth === 4 ? 1.955 : 1.90, .537], [.13, mouth === 3 ? 1.985 : mouth === 1 ? 1.92 : 1.965, .515]], .018);
  if (ch.facialHair > 0) {
    if ([1, 3, 4, 5].includes(Number(ch.facialHair))) { const beard = oval(hair, 0, Number(ch.facialHair)===4?1.72:1.77, .41, Number(ch.facialHair) === 5 ? .13 : Number(ch.facialHair)===4?.33:.30, Number(ch.facialHair)===4?.22:.15, Number(ch.facialHair)===4?.14:.10); if (Number(ch.facialHair) === 1) { beard.material = hair.clone(); beard.material.transparent = true; beard.material.opacity = .25; } }
    if(Number(ch.facialHair)===6)for(const sign of [-1,1])tube(hair,[[sign*.51,2.15,.26],[sign*.50,1.99,.30],[sign*.43,1.86,.34]],.04);
    if(Number(ch.facialHair)===7)oval(hair,0,1.78,.45,.075,.065,.027);
    if ([2, 3, 4].includes(Number(ch.facialHair))) tube(hair, [[-.16, 1.99, .51], [-.07, 2.01, .56], [0, 1.98, .57], [.07, 2.01, .56], [.16, 1.99, .51]], .032);
  }
  if (ch.glasses > 0) {
    const frame = fitted(material(validColor(ch.glassesColor, "#394957"), .45),'customGlasses'), glasses = Number(ch.glasses);
    for (const sign of glasses === 7 ? [1] : [-1, 1]) {
      ring(frame, sign * .245, 2.19, .595, glasses===9?.22:.18, glasses===11?.10:glasses === 6 ? .17 : .158, .019, root, [2,9,10].includes(glasses) ? "rectangle" : glasses === 3 ? "cat-eye" : glasses === 5 ? "heart" : glasses===12?"star":glasses === 6 ? "aviator" : "round");
      if(glasses===8)tube(frame,[[sign*.245-.17,2.27,.595],[sign*.245,2.34,.605],[sign*.245+.17,2.27,.595]],.027);
      if (glasses === 4 || glasses===10) { const lens = material("#263844", .2); lens.transparent = true; lens.opacity = .87; oval(lens, sign * .245, 2.19, .589, .16, .139, .018); }
    }
    if (glasses !== 7) tube(frame, [[-.065, 2.20, .595], [0, 2.225, .625], [.065, 2.20, .595]], .018);
    for (const sign of [-1, 1]) tube(frame, [[sign * .42, 2.21, .59], [sign * .59, 2.23, .12]], .017);
    if (glasses === 7) tube(frame, [[.39, 2.11, .58], [.49, 1.84, .35], [.53, 1.69, .08]], .007);
  }
  const jewelry = fitted(material(validColor(ch.necklaceColor, "#c99d48"), .28),'customNecklace'); jewelry.metalness = .5;
  if (ch.necklace > 0 && ch.necklace<=5) { tube(jewelry, [[-.16, 1.51, .15], [-.12, 1.29, .239], [0, 1.23, .26], [.12, 1.29, .239], [.16, 1.51, .15]], Number(ch.necklace) === 4 ? .025 : .012); if ([2, 5].includes(Number(ch.necklace))) oval(jewelry, 0, 1.20, .27, .042, .05, .018); if (Number(ch.necklace) === 3) for (let i = -2; i <= 2; i++) oval(jewelry, i * .053, 1.24 + Math.abs(i) * .03, .26, .025, .025, .025); }
  if (ch.earring > 0 && ch.earring<=3) { const metal = fitted(material(validColor(ch.earringColor, "#c99d48"), .28),'customEarring'); metal.metalness = .5; for (const sign of [-1, 1]) { oval(metal, sign * .69, 2.015, .08, .025, .025, .025); if (Number(ch.earring) === 2) oval(metal,sign*.69,1.95,.075,.025,.065,.02); if (Number(ch.earring) === 3) ring(metal, sign * .69, 1.96, .075, .045, .07, .012); } }
  if(ch.customAccessory && textures.customAccessory){
    const accessory=fitted(material('#ffffff'),'customAccessory'), shape=ch.customAccessory.baseStyle;
    if(shape===1){oval(accessory,0,1.1,-.31,.29,.35,.15);for(const sign of [-1,1])tube(accessory,[[sign*.25,1.47,-.15],[sign*.3,1.3,.19],[sign*.24,.88,-.12]],.025);}
    else if(shape===2){for(const sign of [-1,1]){const wing=oval(accessory,sign*.46,1.34,-.28,.34,.41,.08);wing.rotation.z=sign*-.42;}}
    else if(shape===3)tube(accessory,[[0,.66,-.12],[.12,.55,-.45],[.4,.75,-.55],[.45,.9,-.48]],.065);
    else oval(accessory,.12,1.2,.247,.085,.095,.02);
  }
  if (ch.hat > 0) {
    const hat = fitted(material(validColor(ch.hatColor, "#739db8")),'customHat'), number = Number(ch.hat);
    if ([1, 2, 8].includes(number)) { mesh(hairCap, hat, [0, 2.36, -.02], [.70, .56, .61]); if (number === 1) oval(hat, 0, 2.94, -.02, .10, .10, .10); if (number === 2) oval(hat, 0, 2.59, .54, .53, .035, .29); if (number === 8) mesh(new THREE.CylinderGeometry(.69, .78, .08, 48), hat, [0, 2.59, -.01], [1, 1, .91], root, true); }
    else if (number === 3) { mesh(new THREE.CylinderGeometry(.52, .56, .31, 40), hat, [0, 2.85, -.04], [1, 1, .8], root, true); oval(hat, 0, 2.69, -.01, .88, .045, .70); }
    else if (number === 4) { limb(hat, 0, 2.83, -.05, .45, .28, .42); mesh(box, hat, [0, 3.0, -.03], [1.20, .065, 1.05]); tube(jewelry, [[0, 3.04, 0], [.55, 3.04, .13], [.60, 2.66, .18]], .014); }
    else if (number === 5) { const crown=ch.customHat?hat:jewelry; mesh(new THREE.CylinderGeometry(.57, .57, .17, 40, 1, true), crown, [0, 2.78, -.01], [1, 1, .9], root, true); for (let i = 0; i < 6; i++) { const a = i / 6 * Math.PI * 2; mesh(new THREE.ConeGeometry(.095, .25, 4), crown, [Math.cos(a) * .54, 2.95, Math.sin(a) * .48], [1, 1, 1], root, true); } }
    else if (number === 6) { mesh(new THREE.ConeGeometry(.36, .77, 40), hat, [0, 3.01, -.03], [1, 1, .9], root, true); oval(white, 0, 3.4, -.03, .07, .07, .07); }
    else if (number === 7) tube(hat, [[-.65, 2.18, .02], [-.55, 2.65, -.02], [0, 2.81, -.04], [.55, 2.65, -.02], [.65, 2.18, .02]], .055);
  }
  addAvatarDetails({ch,root,mesh,oval,limb,tube,ring,material,white,ink,hair,jewelry});
  root.userData.dispose = () => { owned.forEach(geometry => geometry.dispose()); const materials = new Set(createdMaterials); root.traverse(item => { if (item.material) materials.add(item.material); }); materials.forEach(value => value.dispose()); };
  return root;
}

let studio;
function getStudio() {
  if (studio) return studio;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1); renderer.setSize(256, 256); renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.1;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb2c2d5, 2.4));
  const key = new THREE.DirectionalLight(0xfff4e6, 3.5); key.position.set(-3, 5, 5); key.castShadow = true; key.shadow.mapSize.set(1024, 1024); key.shadow.camera.left = -2; key.shadow.camera.right = 2; key.shadow.camera.top = 3.8; key.shadow.camera.bottom = -.5; key.shadow.normalBias = .025; scene.add(key);
  const fill = new THREE.DirectionalLight(0xc9e8ff, 1.2); fill.position.set(4, 3, -3); scene.add(fill);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.ShadowMaterial({ color: 0x26384b, opacity: .12 })); floor.rotation.x = -Math.PI / 2; floor.position.y = .015; floor.receiveShadow = true; scene.add(floor);
  const camera = new THREE.OrthographicCamera(-1.5, 1.5, 1.5, -1.5, .1, 30);
  studio = { renderer, scene, camera, floor }; return studio;
}

const textureCache = new Map();
async function itemTextures(character) {
  const result={};
  for(const field of ['customClothing','customHair','customBottom','customShoes','customHat','customGlasses','customNecklace','customEarring','customAccessory']) {
    const source=character[field]?.texture;
    if(!/^data:image\/(?:png|jpeg|webp);base64,[a-z0-9+/=]+$/i.test(source||'')||source.length>100000)continue;
    if(!textureCache.has(source)){const image=new Image();image.src=source;await image.decode();const texture=new THREE.Texture(image);texture.colorSpace=THREE.SRGBColorSpace;texture.needsUpdate=true;textureCache.set(source,texture);if(textureCache.size>40){const key=textureCache.keys().next().value;textureCache.get(key).dispose();textureCache.delete(key);}}
    result[field]=textureCache.get(source);
  }
  return result;
}
export async function renderCharacter(character, { framing = "portrait", yaw = -.10, size = 256 } = {}) {
  const textures=await itemTextures(character);
  const { renderer, scene, camera } = getStudio(), model = createCharacter(character,textures);
  const color = new THREE.Color(validColor(character.bg, "#e6f2fb")); scene.background = color;
  model.rotation.y = yaw; scene.add(model);
  const bounds=new THREE.Box3().setFromObject(model),height=bounds.max.y-bounds.min.y,width=bounds.max.x-bounds.min.x;
  const full=framing==="full",span=full?Math.max(1.72,height*.56,width*.56):Math.max(1,(bounds.max.y-1.45)*.57,width*.54),target=full?(bounds.min.y+bounds.max.y)/2:(1.45+bounds.max.y)/2;
  camera.left = -span; camera.right = span; camera.top = span; camera.bottom = -span;
  camera.position.set(2.1, target + .85, 8.4); camera.lookAt(0, target, 0); camera.updateProjectionMatrix(); renderer.setSize(size, size);
  try { renderer.render(scene, camera); return renderer.domElement.toDataURL("image/png"); }
  finally { scene.remove(model); model.userData.dispose(); renderer.renderLists.dispose(); }
}

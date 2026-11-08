import * as THREE from 'three';

// Rounded fabric contours: a hem, shaped waist, shoulders and an open neckline.
// IDs in avatarOptions are append-only so existing saved outfits keep their meaning.
const baseProfile = [[.79,.305,.21],[.81,.34,.233],[.87,.335,.23],[1.02,.315,.218],[1.23,.342,.23],[1.35,.33,.214],[1.41,.275,.18],[1.46,.145,.121],[1.48,.13,.115]];
const segments = 40;
export function fabricGeometry(profile, { centerX=0, centerZ=0, pleats=0 } = {}) {
  const positions=[],uv=[],indices=[],minY=profile[0][0],maxY=profile.at(-1)[0],maxWidth=Math.max(...profile.map(row=>row[1]));
  for(let row=0;row<profile.length;row++) {
    const [y,width,depth]=profile[row];
    for(let col=0;col<=segments;col++) {
      const a=col/segments*Math.PI*2, fold=1+pleats*Math.cos(a*10), x=Math.sign(Math.sin(a))*Math.abs(Math.sin(a))**.7*width*fold;
      positions.push(centerX+x,y,centerZ+Math.sign(Math.cos(a))*Math.abs(Math.cos(a))**.7*depth*fold);
      uv.push(.5+x/(2*maxWidth),(y-minY)/(maxY-minY));
      if(row&&col) {const b=row*(segments+1)+col;indices.push(b,b-1,b-segments-2,b,b-segments-2,b-segments-1);}
    }
  }
  // Cap the bottom; leave the neck opening under the character's neck.
  const cap=positions.length/3;positions.push(centerX,minY,centerZ);uv.push(.5,0);
  for(let col=0;col<segments;col++)indices.push(cap,col+1,col);
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));geometry.setIndex(indices);geometry.computeVertexNormals();return geometry;
}

function sleeveGeometry(sign,long,puff=false,cuff=false,flared=false) {
  const rings=cuff?[.87,.94,1]:[0,.08,.28,.55,.84,.96,1],positions=[],uv=[],indices=[];
  const start=new THREE.Vector3(sign*.285,1.365,0),end=new THREE.Vector3(sign*(long?.443:.45),long?.94:1.17,.012),direction=end.clone().sub(start).normalize();
  const side=new THREE.Vector3(-direction.y,direction.x,0),depth=new THREE.Vector3(0,0,1);
  for(let row=0;row<rings.length;row++)for(let col=0;col<=28;col++) {
    const t=rings[row],angle=col/28*Math.PI*2,center=start.clone().lerp(end,t);
    const radius=(.134-.027*t+(puff?.045*Math.sin(t*Math.PI):0))*(t===1?.94:1)+(cuff?.005:0)+(flared?.05*t:0);
    const p=center.addScaledVector(side,Math.cos(angle)*radius).addScaledVector(depth,Math.sin(angle)*radius*.95);
    positions.push(...p);uv.push(col/28,t);
    if(row&&col){const b=row*29+col;indices.push(b,b-1,b-30,b,b-30,b-29);}
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));geometry.setIndex(indices);geometry.computeVertexNormals();return geometry;
}

export function addWardrobe({ch,root,mesh,oval,limb,tube,material,skin,shirt,pants,shoes,ink}) {
  const top=Number(ch.topStyle)||0, bottom=Number(ch.pantsStyle)||0, shoe=Number(ch.shoeStyle)||0;
  const trim=material(/^#[\da-f]{6}$/i.test(ch.outfitAccent||'')?ch.outfitAccent:'#fffaf1');
  const seam=material(shirt.color.clone().multiplyScalar(.72)), stitch=material(shirt.color.clone().lerp(new THREE.Color('#ffffff'),.42));
  const dress=[9,13,30,35].includes(top), sleeveless=[3,9,27].includes(top), shortSleeve=[0,5,6,7,8,11,18,20,21,24,25,26,30,33,34].includes(top);
  const loose=[1,16,19,22,23,28,32,35].includes(top), skirt=bottom===4;
  const horizontal=(mat,x,y,z,rx,rz,r=.012)=>tube(mat,Array.from({length:41},(_,i)=>{const a=i/40*Math.PI*2;return[x+Math.sin(a)*rx,y,z+Math.cos(a)*rz];}),r,root,true);
  const longPants=ch.customBottom?bottom!==1:bottom===0?[2,4,10,12,14,15,22,29,32,35].includes(top):bottom!==1;
  if(!dress||bottom!==0) {
    if(skirt)mesh(fabricGeometry([[.48,.37,.24],[.51,.38,.25],[.76,.275,.20],[.88,.27,.19]],{pleats:.035}),pants,[0,0,0],[1,1,1],root,true);
    else for(const sign of [-1,1]){
      const width=bottom===7?.15:.117,low=longPants?.22:.55;
      const leg=mesh(fabricGeometry([[low,width*.94,.115],[low+.025,width,.12],[.68,.14,.17],[.78,.142,.17]],{centerX:sign*.15}),pants,[0,0,0],[1,1,1],root,true);leg.userData.avatarPart='trouser-leg';
      if(bottom===2)tube(stitch,[[sign*.25,.25,.045],[sign*.255,.50,.09],[sign*.265,.72,.11]],.005);
      if(bottom===3)horizontal(trim,sign*.15,.25,0,.115,.12,.012);
      if(bottom===5){const pocket=mesh(new THREE.BoxGeometry(.065,.16,.10),pants,[sign*.285,.58,.06],[1,1,1],root,true);pocket.rotation.z=sign*.06;}
    }
  }
  for(const sign of [-1,1]) {
    limb(skin,sign*.16,.39,.015,.10,.47,.10);
    const x=sign*.17,sole=material(shoes.color.clone().multiplyScalar(.72)),high=[1,4,6].includes(shoe);
    const soleProfile=[[.025,.13,.205],[.04,.15,.23],[.07,.15,.23]];
    mesh(fabricGeometry(soleProfile,{centerX:x,centerZ:.09}),sole,[0,0,0],[1,1,1],root,true);
    if(shoe===3){for(const z of [.12,.24])tube(shoes,[[x-.11,.09,z],[x,.145,z+.02],[x+.11,.09,z]],.025);}
    else {
      const profile=[[.065,.143,.22],[.10,.145,.219],[.145,.13,.19],[.19,.095,.12],...(high?[[shoe===6?.39:.30,.11,.12]]:[])];
      mesh(fabricGeometry(profile,{centerX:x,centerZ:.07}),shoes,[0,0,0],[1,1,1],root,true);
      if(high)horizontal(trim,x,shoe===6?.38:.29,.07,.108,.12,.012);
      if([0,4,9].includes(shoe))for(let i=0;i<3;i++)tube(trim,[[x-.065,.163+i*.006,.15+i*.025],[x+.065,.163+i*.006,.15+i*.025]],.008);
      if([2,5,8].includes(shoe))tube(trim,[[x-.095,.125,.20],[x,.17,.22],[x+.095,.125,.20]],shoe===5?.018:.009);
      if(shoe===7)oval(trim,x,.14,.22,.11,.055,.07);
      if(shoe===8)tube(trim,[[x-.10,.13,.12],[x-.10,.14,.25],[x,.15,.28],[x+.10,.14,.25],[x+.10,.13,.12]],.006);
      if(shoe===9){shoes.roughness=.44;tube(sole,[[x-.10,.12,.24],[x,.145,.27],[x+.10,.12,.24]],.009);}
    }
  }

  const profile=baseProfile.map(([y,w,d])=>[y,w*(loose?1.065:1),d*(loose?1.04:1)]);
  if(dress){profile[0]=[.49,.425,.29];profile[1]=[.52,.43,.29];profile[2]=[.75,.36,.24];}
  if(top===26){profile[0]=[.72,.43,.25];profile[1]=[.75,.42,.25];profile[2]=[.90,.29,.205];}
  const body=mesh(fabricGeometry(profile),shirt,[0,0,0],[1,1,1],root,true);body.userData.avatarPart='shirt-body';body.name='Tailored shirt';
  function surface(x,y,extra=.009){let lower=profile[0],upper=profile.at(-1);for(let i=1;i<profile.length;i++)if(profile[i][0]>=y){lower=profile[i-1];upper=profile[i];break;}const t=Math.max(0,Math.min(1,(y-lower[0])/(upper[0]-lower[0]||1))),w=THREE.MathUtils.lerp(lower[1],upper[1],t),d=THREE.MathUtils.lerp(lower[2],upper[2],t);return d*Math.max(.02,1-(Math.abs(x)/w)**(2/.7))**(.7/2)+extra;}
  const frontTube=(mat,points,r=.007)=>tube(mat,points.map(([x,y])=>[x,y,surface(x,y)]),r);
  function patch(mat,x,y,w,h,kind='pocket'){
    const shape=new THREE.Shape(),r=.018;shape.moveTo(x-w/2+r,y-h/2);shape.lineTo(x+w/2-r,y-h/2);shape.quadraticCurveTo(x+w/2,y-h/2,x+w/2,y-h/2+r);shape.lineTo(x+w/2,y+h/2);shape.lineTo(x-w/2,y+h/2);shape.lineTo(x-w/2,y-h/2+r);shape.quadraticCurveTo(x-w/2,y-h/2,x-w/2+r,y-h/2);
    const geometry=new THREE.ShapeGeometry(shape,8),p=geometry.attributes.position;for(let i=0;i<p.count;i++)p.setZ(i,surface(p.getX(i),p.getY(i),.014));geometry.computeVertexNormals();const item=mesh(geometry,mat,[0,0,0],[1,1,1],root,true);item.userData.avatarPart=kind;
    frontTube(stitch,[[x-w/2+.012,y+h/2-.012],[x+w/2-.012,y+h/2-.012]],.004);return item;
  }
  const buttons=(x,ys)=>ys.forEach(y=>oval(trim,x,y,surface(x,y,.019),.013,.013,.008));
  function collar(wide=false){for(const sign of [-1,1]){const shape=new THREE.Shape();shape.moveTo(sign*.04,1.45);shape.lineTo(sign*(wide?.24:.19),1.435);shape.lineTo(sign*(wide?.19:.13),wide?1.26:1.34);shape.lineTo(sign*.015,1.405);shape.closePath();const geometry=new THREE.ShapeGeometry(shape),p=geometry.attributes.position;for(let i=0;i<p.count;i++)p.setZ(i,surface(p.getX(i),p.getY(i),.027));geometry.computeVertexNormals();mesh(geometry,trim,[0,0,0],[1,1,1],root,true);}}
  for(const sign of [-1,1]){
    const arm=limb(skin,sign*.405,1.10,-.005,.085,.49,.09);arm.rotation.z=sign*.18;
    oval(skin,sign*.445,.85,.015,.105,.12,.105);oval(skin,sign*.375,.86,.085,.045,.06,.055);
    if(!sleeveless){const long=!shortSleeve,sleeve=mesh(sleeveGeometry(sign,long,top===25,false,top===23),[12,21,24,30,31].includes(top)?trim:shirt,[0,0,0],[1,1,1],root,true);sleeve.userData.avatarPart='shirt-sleeve';mesh(sleeveGeometry(sign,long,top===25,true,top===23),[1,7,12,16,21,28,31,32].includes(top)?trim:seam,[0,0,0],[1,1,1],root,true);}
  }
  limb(skin,0,1.56,0,.115,.25,.12);
  const hemY=profile[1][0]+.006,hemW=profile[1][1],hemD=profile[1][2];
  tube(stitch,Array.from({length:41},(_,i)=>{const a=i/40*Math.PI*2;return[Math.sin(a)*hemW,hemY,Math.cos(a)*hemD];}),.006,root,true);
  horizontal([1,16,28,31].includes(top)?trim:seam,0,1.475,.015,.137,.123,.013);
  for(const sign of [-1,1])frontTube(seam,[[sign*.302,.85],[sign*.31,1.15],[sign*.29,1.34]],.004);

  if([1,22].includes(top)){
    const hoodProfile=[[1.36,.19,.15],[1.43,.245,.19],[1.57,.25,.19],[1.64,.18,.16],[1.67,.10,.11]];
    mesh(fabricGeometry(hoodProfile,{centerZ:-.13}),shirt,[0,0,0],[1,1,1],root,true);
    for(const sign of [-1,1])frontTube(trim,[[sign*.115,1.43],[sign*.13,1.25]],.009);
    patch(shirt,0,.99,.31,.16);frontTube(seam,[[-.15,1.06],[-.17,.96],[0,.93],[.17,.96],[.15,1.06]],.008);
  }
  if([2,4,6,10,15,17,18,19,22,29,33,35].includes(top)){
    collar([4,10,19,35].includes(top));frontTube(seam,[[0,top===6?1.22:.84],[0,1.39]],.007);buttons(0,top===6?[1.24,1.33]:[.94,1.07,1.20,1.31]);
    if(top===17)patch(shirt,.16,1.22,.13,.14);
    if([2,10,15,18,22,29,35].includes(top))for(const sign of [-1,1])patch(shirt,sign*.18,top===15?1.19:1.01,.14,.15);
    if([4,10].includes(top)){patch(trim,0,1.29,.10,.26,'shirt-insert');frontTube(ink,[[0,1.4],[.027,1.29],[0,1.15],[-.027,1.29],[0,1.4]],.019);}
    if(top===19)patch(trim,0,1.24,.10,.30,'cardigan-insert');
    if(top===29)patch(shirt,.16,1.25,.13,.13);
  }
  if([7,8,12,21,24,32,33].includes(top)){
    const stripes=top===8?[.95,1.08,1.21,1.34]:top===7?[.89,1.30]:[1.32];
    for(const y of stripes)frontTube(trim,[[-.29,y],[-.16,y],[0,y],[.16,y],[.29,y]],top===8?.016:.012);
    if(top===7){frontTube(trim,[[.02,1.22],[-.05,1.22],[-.05,1.11],[.03,1.11],[.03,1.04],[-.05,1.04]],.012);}
    if(top===12)frontTube(trim,[[.18,1.27],[.10,1.27],[.08,1.21],[.10,1.16],[.18,1.16]],.014);
    if(top===24){collar(true);frontTube(trim,[[-.07,1.29],[0,1.22],[.07,1.29]],.02);}
    if(top===33)for(const sign of [-1,1])patch(trim,sign*.23,1.14,.10,.32,'contrast-panel');
  }
  if([5,13,23].includes(top)){
    frontTube(trim,[[-.20,1.43],[.19,1.02]],.016);frontTube(trim,[[-.29,.96],[0,.92],[.29,.96]],.018);
    for(const sign of [-1,1])frontTube(trim,[[0,.94],[sign*.09,.92],[sign*.12,.86],[sign*.03,.90],[0,.94]],.015);
  }
  if(top===14)mesh(fabricGeometry([[1.43,.16,.14],[1.59,.15,.13]]),shirt,[0,0,0],[1,1,1],root,true);
  if([11,28].includes(top)){
    const ys=top===28?[1.03,1.23]:[1.20];for(const y of ys)for(const x of top===28?[-.17,0,.17]:[0]){
      const shape=new THREE.Shape();for(let i=0;i<10;i++){const a=i/10*Math.PI*2+Math.PI/2,r=i%2?.026:.06,px=x+Math.cos(a)*r,py=y+Math.sin(a)*r;if(i)shape.lineTo(px,py);else shape.moveTo(px,py);}shape.closePath();const g=new THREE.ShapeGeometry(shape);g.translate(0,0,surface(x,y,.014));mesh(g,trim,[0,0,0],[1,1,1],root,true);
    }
  }
  if(top===16)for(const x of [-.18,-.06,.06,.18])frontTube(stitch,Array.from({length:15},(_,i)=>[x+Math.sin(i*Math.PI/2)*.018,.88+i*.032]),.006);
  if([20,30].includes(top)||bottom===6){
    const bibColor=top===30?shirt:pants;patch(bibColor,0,1.09,.32,.32,'dungaree-bib');for(const sign of [-1,1]){frontTube(bibColor,[[sign*.19,1.43],[sign*.16,1.19]],.033);buttons(sign*.15,[1.19]);}patch(bibColor,0,1.11,.16,.12);
  }
  if([25,34].includes(top))for(let i=0;i<9;i++){const a=Math.PI+i/8*Math.PI;oval(trim,Math.cos(a)*.16,1.45+Math.sin(a)*.08,.14+Math.sin(-a)*.08,.038,.027,.012);}
  if(top===27){frontTube(trim,[[0,.83],[0,1.43]],.013);patch(seam,.15,1.20,.12,.13);}
  if(top===31){frontTube(trim,[[-.15,1.45],[0,1.23],[.15,1.45]],.025);for(let y=.94;y<1.23;y+=.10)frontTube(stitch,[[-.18,y],[0,y-.04],[.18,y]],.005);}
}

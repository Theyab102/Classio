import * as THREE from 'three';
import {fabricGeometry} from './avatarWardrobe';
const valid=(value,fallback)=>/^#[\da-f]{6}$/i.test(value||'')?value:fallback;

export function addAvatarDetails({ch,root,mesh,oval,tube,ring,material,white,ink,jewelry}) {
  const hat=material(valid(ch.hatColor,'#739db8')),accent=material(valid(ch.accessoryColor,'#b66957'));
  const trim=material(valid(ch.outfitAccent,'#fffaf1')),seam=material(hat.color.clone().multiplyScalar(.75));
  const horizontal=(mat,x,y,z,rx,rz,r=.012)=>tube(mat,Array.from({length:41},(_,i)=>{const a=i/40*Math.PI*2;return[x+Math.sin(a)*rx,y,z+Math.cos(a)*rz];}),r,root,true);
  const fabric=(profile,mat=hat,options={})=>mesh(fabricGeometry(profile,options),mat,[0,0,0],[1,1,1],root,true);
  function star(mat,x,y,z,r=.05){const shape=new THREE.Shape();for(let i=0;i<10;i++){const a=i/10*Math.PI*2+Math.PI/2,rad=i%2?r*.48:r;if(i)shape.lineTo(Math.cos(a)*rad,Math.sin(a)*rad);else shape.moveTo(Math.cos(a)*rad,Math.sin(a)*rad);}shape.closePath();return mesh(new THREE.ExtrudeGeometry(shape,{depth:.014,bevelEnabled:true,bevelThickness:.003,bevelSize:.003,bevelSegments:2,steps:1}),mat,[x,y,z],[1,1,1],root,true);}
  const number=Number(ch.hat)||0;
  if(number===9){const beret=oval(hat,.07,2.82,-.04,.68,.17,.57);beret.rotation.z=.13;horizontal(seam,0,2.70,-.02,.58,.49,.025);oval(seam,.08,2.99,-.03,.035,.04,.035);}
  if(number===10){fabric([[2.65,.59,.50],[2.70,.66,.55],[2.87,.52,.44],[2.99,.08,.08]]);oval(hat,0,2.66,.47,.51,.025,.28);for(let i=0;i<6;i++){const a=i/6*Math.PI*2;tube(seam,[[Math.sin(a)*.59,2.68,Math.cos(a)*.5],[Math.sin(a)*.46,2.89,Math.cos(a)*.39],[0,2.99,0]],.006);}oval(seam,0,3.0,0,.04,.016,.04);}
  if(number===11){fabric([[2.64,.84,.69],[2.67,.88,.72],[2.70,.52,.45]],hat,{pleats:.012});fabric([[2.68,.52,.45],[2.94,.43,.38],[3.0,.35,.30]]);horizontal(accent,0,2.73,0,.515,.445,.025);}
  if([12,15].includes(number)){
    fabric([[2.58,.65,.57],[2.63,.69,.60],[2.79,.60,.53],[3.0,.36,.32],[3.06,.10,.09]]);horizontal(trim,0,2.64,0,.67,.58,.035);
    if(number===12)for(let i=0;i<12;i++){const a=i/12*Math.PI*2;tube(seam,[[Math.sin(a)*.64,2.67,Math.cos(a)*.56],[Math.sin(a)*.55,2.86,Math.cos(a)*.48],[Math.sin(a)*.28,3.02,Math.cos(a)*.24]],.009);}
    if(number===15)for(const sign of [-1,1]){fabric([[2.03,.09,.105],[2.13,.12,.12],[2.61,.11,.11]],hat,{centerX:sign*.61});tube(trim,[[sign*.61,2.10,.09],[sign*.61,1.91,.10],[sign*.59,1.85,.11]],.016);oval(trim,sign*.59,1.85,.11,.04,.05,.04);}
  }
  if(number===13){for(const sign of [-1,1]){const bow=oval(hat,sign*.16,2.77,.34,.20,.12,.06);bow.rotation.z=sign*.35;}oval(seam,0,2.77,.40,.06,.07,.04);}
  if(number===14){horizontal(hat,0,2.57,0,.65,.57,.045);oval(hat,0,2.57,.53,.53,.03,.30);}
  if(number===16){horizontal(accent,0,2.63,0,.59,.51,.018);for(let i=0;i<7;i++){const a=-Math.PI/2+i/6*Math.PI,x=Math.sin(a)*.58,z=Math.cos(a)*.51;for(let p=0;p<5;p++){const theta=p/5*Math.PI*2;oval(hat,x+Math.cos(theta)*.045,2.66+Math.sin(theta)*.045,z+.015,.036,.036,.025);}oval(trim,x,2.66,z+.04,.028,.028,.018);}}
  if(number===17){fabric([[2.65,.70,.59],[2.70,.71,.60],[2.72,.47,.41]]);fabric([[2.69,.46,.40],[3.18,.49,.42],[3.22,.48,.41]]);horizontal(accent,0,2.80,0,.47,.41,.034);}

  const necklace=Number(ch.necklace)||0;
  if(necklace===5)for(let i=0;i<9;i++){const t=i/8*Math.PI;oval(white,Math.cos(t)*.16,1.43-Math.sin(t)*.19,.17+Math.sin(t)*.10,.024,.024,.024);}
  if(necklace===6){horizontal(accent,0,1.49,0,.17,.15,.055);for(const sign of [-1,1])tube(accent,[[sign*.06,1.47,.19],[sign*.10,1.28,.27],[sign*.07,1.10,.26]],.045);}
  if([7,8].includes(necklace)){for(const sign of [-1,1]){const bow=oval(jewelry,sign*.066,1.43,.20,.075,.037,.023);bow.rotation.z=sign*.25;}oval(jewelry,0,1.43,.225,.026,.032,.02);if(necklace===8)for(const sign of [-1,1])tube(jewelry,[[0,1.42,.23],[sign*.08,1.28,.26],[sign*.055,1.24,.25]],.016);}
  if(necklace===9){tube(jewelry,[[-.16,1.48,.15],[0,1.26,.27],[.16,1.48,.15]],.009);tube(jewelry,[[.02,1.25,.28],[-.035,1.24,.28],[-.044,1.19,.28],[.005,1.17,.28],[.025,1.19,.28]],.014);}
  const earring=Number(ch.earring)||0;
  if(earring>=4){const metal=material(valid(ch.earringColor,'#c99d48'));metal.metalness=.5;for(const sign of [-1,1]){
    const x=sign*.69;oval(metal,x,2.015,.08,.018,.018,.018);
    if(earring===4)star(metal,x,1.95,.08,.055);
    if(earring===5){tube(metal,[[x,2.01,.08],[x,1.94,.08]],.008);oval(metal,x,1.91,.08,.035,.06,.02);}
    if(earring===6){tube(metal,[[x,1.99,.08],[x,1.88,.08]],.012);tube(metal,[[x-.036,1.95,.08],[x+.036,1.95,.08]],.012);}
    if(earring===7)tube(metal,[[x+.025,2.0,.08],[x-.025,1.99,.08],[x-.04,1.94,.08],[x-.005,1.91,.08],[x+.025,1.93,.08]],.012);
    if(earring===8){for(let i=0;i<5;i++){const a=i/5*Math.PI*2;oval(metal,x+Math.cos(a)*.032,1.96+Math.sin(a)*.032,.08,.025,.025,.016);}oval(trim,x,1.96,.10,.015,.015,.01);}
    if(earring===9)for(let i=0;i<2;i++)ring(metal,x,1.965-i*.075,.08,.037,.045,.009);
  }}

  if(ch.customAccessory)return;
  const accessory=Number(ch.accessoryStyle)||0;
  if(accessory===1){oval(accent,0,1.13,-.32,.29,.34,.14);oval(accent,0,1.02,-.45,.20,.15,.065);for(const sign of [-1,1])tube(accent,[[sign*.23,1.44,-.16],[sign*.30,1.26,.20],[sign*.25,.91,-.13]],.023);}
  if([2,8].includes(accessory)){const x=accessory===2?.39:.25;oval(accent,x,.93,.15,.20,.19,.10);tube(accent,[[-.23,1.48,.11],[.04,1.19,.27],[x,1.0,.25]],.025);oval(trim,x,.95,.26,.025,.023,.012);}
  if(accessory===3)for(const sign of [-1,1]){const wing=oval(trim,sign*.45,1.30,-.28,.30,.34,.07);wing.rotation.z=sign*-.4;for(let i=0;i<3;i++){const feather=oval(trim,sign*(.36+i*.09),1.11+i*.08,-.25,.08,.17,.04);feather.rotation.z=sign*-.5;}}
  if(accessory===4)tube(accent,[[0,.70,-.15],[.12,.56,-.46],[.44,.68,-.57],[.47,.94,-.49],[.36,1.0,-.43]],.06);
  if([5,6].includes(accessory))for(const sign of accessory===5?[-1]:[-1,1]){horizontal(accent,sign*.44,.91,.012,.09,.095,.02);if(accessory===5){oval(ink,sign*.44,.92,.109,.046,.043,.012);tube(trim,[[sign*.44,.92,.125],[sign*.44+.023,.935,.125]],.004);}}
  if(accessory===7)star(trim,.15,1.21,.245,.065);
}

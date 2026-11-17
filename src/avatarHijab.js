import * as THREE from 'three';
export function addHijab({ch,root,mesh,tube,oval,material}){
  const style=Number(ch.hijabStyle)||0;if(!style)return;
  const fabric=material(/^#[\da-f]{6}$/i.test(ch.hijabColor||'')?ch.hijabColor:'#a46378'),seam=material(new THREE.Color(fabric.color).multiplyScalar(.82));
  const rings=[{x:.51,y:.50,z:.43},{x:.71,y:.70,z:.06},{x:.58,y:.62,z:-.40},{x:.02,y:.03,z:-.63}],positions=[],indices=[],steps=64;
  for(const ring of rings)for(let i=0;i<=steps;i++){const angle=i/steps*Math.PI*2;positions.push(Math.sin(angle)*ring.x,2.20+Math.cos(angle)*ring.y,ring.z);}
  for(let row=0;row<rings.length-1;row++)for(let i=0;i<steps;i++){const at=row*(steps+1)+i;indices.push(at,at+1,at+steps+1,at+1,at+steps+2,at+steps+1);}
  const hood=new THREE.BufferGeometry();hood.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));hood.setIndex(indices);hood.computeVertexNormals();fabric.side=THREE.DoubleSide;
  mesh(hood,fabric,[0,0,0],[1,1,1],root,true).userData.avatarPart='hijab';
  tube(seam,Array.from({length:49},(_,i)=>{const a=i/48*Math.PI*2;return[Math.sin(a)*.514,2.20+Math.cos(a)*.503,.437];}),.012,root,true);
  // Shoulder/neck cloth remains outside the original head and shirt.
  const lower=style===3?1.47:style===4?1.03:1.24,width=style===3?.37:style===4?.76:.61;
  const drape=new THREE.LatheGeometry([new THREE.Vector2(width,lower),new THREE.Vector2(width*.94,lower+.14),new THREE.Vector2(.40,1.66),new THREE.Vector2(.43,1.81)],56);drape.scale(1,1,.69);
  mesh(drape,fabric,[0,0,.02],[1,1,1],root,true).userData.avatarPart='hijab-drape';
  if(style===2){for(let i=0;i<3;i++){const blend=.5+i*.13;tube(seam,Array.from({length:25},(_,step)=>{const angle=-Math.PI/2+step/24*Math.PI;return[Math.sin(angle)*(.51+.20*blend),2.20+Math.cos(angle)*(.50+.20*blend),.437-.37*blend];}),.012);}}
  if(style===1||style===4){for(let i=0;i<(style===4?4:2);i++)tube(seam,[[-width*.65,1.62,.30],[0,1.35-i*.055,.42],[width*.73,lower+.07+i*.035,.27]],.009);}
  if(style===4)oval(fabric,-.43,1.08,.24,.16,.31,.075);
}

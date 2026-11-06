import { memo, useId } from 'react';

export const avatarDefaults = { portraitStyle:0, eyeSize:1, noseSize:1, mouthSize:1, browHeight:0, skin:'#edc19c', hair:'#3d2b24', hairStyle:1, eyes:'#476775', top:'#2b6777', bg:'#c8d8e4', mouth:0, eyebrow:0, eyeShape:0, faceShape:0, nose:0, eyeSpacing:0, mouthPosition:0, topStyle:0, blush:false, lips:false, freckles:false, lipColor:'#b96669', hat:0, hatColor:'#2b6777', glasses:0, glassesColor:'#31434a', facialHair:0, necklace:0, necklaceColor:'#d4ad62', earring:0, earringColor:'#d4ad62', name:'' };
const colour = (value, fallback) => /^#[0-9a-f]{3}(?:[0-9a-f]{3})?$/i.test(value || '') ? value : fallback;
function shade(hex, amount) {
  const full = hex.length === 4 ? '#' + [...hex.slice(1)].map(value => value + value).join('') : hex;
  return '#' + [1,3,5].map(index => Math.max(0,Math.min(255,parseInt(full.slice(index,index+2),16)+amount)).toString(16).padStart(2,'0')).join('');
}
const faces = [
  'M51 78C50 43 67 28 100 28s50 15 49 50l-3 29c-3 32-23 51-46 51s-43-19-46-51Z',
  'M56 77c0-33 15-51 44-51s44 18 44 51l-2 34c-2 31-22 52-42 52s-40-21-42-52Z',
  'M50 77c0-33 18-49 50-49s50 16 50 49l-2 42c-1 23-15 36-48 36s-47-13-48-36Z',
  'M50 76c0-31 19-49 50-49s50 18 50 49c0 36-19 66-50 88-31-22-50-52-50-88Z',
  'M58 76c0-36 14-51 42-51s42 15 42 51v34c0 34-18 56-42 56s-42-22-42-56Z',
  'M45 79c0-31 20-48 55-48s55 17 55 48l-2 29c-3 28-24 47-53 47s-50-19-53-47Z',
];
const hairFront = [
  'M51 77C42 41 64 20 100 20s58 21 49 57l-7-22c-21-14-63-14-84 0Z',
  'M48 82C38 46 54 17 95 17c38-5 64 18 57 64l-13-29c-16 9-28 2-38-12-14 20-31 25-46 22Z',
  'M47 83C37 42 65 18 101 18c39 0 60 25 51 65l-12-26-15 9-9-17-14 19-15-18-15 15-14-8Z',
  'M47 90C36 67 42 30 74 24c30-19 74-4 80 35l-3 32-12-32c-19 12-40-12-44-18-9 17-23 20-35 20Z',
  'M48 104C34 85 38 38 66 25c27-17 65-9 80 16l6 62-13-42c-25-5-29-22-31-26-10 18-25 28-47 30Z',
  'M47 98C32 74 43 33 68 24c32-16 68-5 82 23l4 49-15-36c-17 6-31-16-36-23-7 22-29 18-42 29Z',
  'M49 79C39 44 61 21 99 21s62 23 52 58l-13-25c-28-2-54-5-65-16L58 60Z',
  'M49 79C40 41 62 26 100 26s60 15 51 53l-11-24c-27-3-53-4-65-17L58 60Z',
  'M45 83C27 72 29 37 54 27c3-20 34-29 50-18 18-13 47 0 47 17 23 5 31 35 6 58l-13-32c-31 16-61 4-81 2Z',
  'M45 83C38 41 62 20 100 20s62 21 55 63l-15-27c-20 10-59 10-80 0Z',
  'M47 81C34 65 42 28 69 24c17-18 49-13 60 1 29-3 40 37 23 57l-12-25c-27 8-50 5-78 0Z',
  'M46 85C38 48 60 21 100 21s62 27 54 64l-15-27c-24 8-52 7-78 0Z',
  'M48 82C43 60 50 41 65 35l14-18 8 9 12-20 12 18 11-6 14 20c16 8 21 29 15 45l-13-27c-23 2-51 2-75 0Z',
  'M50 80C42 42 63 25 100 25s58 17 50 55l-11-26c-25-6-54-6-78 0Z',
  'M50 81c-4-26 9-42 25-49L88 9l12 10 13-13 13 30c17 8 27 25 24 45l-13-24c-24-4-50-3-74 1Z',
  'M47 85C38 46 62 21 100 21s62 25 53 64l-13-26c-25 5-54 5-80 0Z',
  'M49 79C40 42 62 19 100 19s60 23 51 60l-12-28c-21 7-56 7-78 0Z',
];

function Avatar({ character = {}, size = 80, animate = false, label }) {
  const id = 'classio-avatar-' + useId().replace(/[^a-zA-Z0-9]/g,'');
  const ch = { ...avatarDefaults, ...character };
  for (const key of ['skin','hair','eyes','top','bg','hatColor','glassesColor','necklaceColor','earringColor','lipColor']) ch[key] = colour(ch[key],avatarDefaults[key]);
  const classic=Number(ch.portraitStyle)!==1;
  const featureScale=key=>Math.max(0.65,Math.min(1.4,Number(ch[key])||1));
  const skinShadow = shade(ch.skin,-27), hairShadow = shade(ch.hair,-24);
  const hs = Math.abs(Number(ch.hairStyle) || 0) % 17, eye = Math.abs(Number(ch.eyeShape) || 0) % 5;
  const facial = Number(ch.facialHair) || 0, top = Number(ch.topStyle) || 0, glasses = Number(ch.glasses) || 0, hat = Number(ch.hat) || 0;
  const eyeSpacing = Math.max(-6,Math.min(6,Number(ch.eyeSpacing) || 0));
  const mouthY = Math.max(-4,Math.min(4,Number(ch.mouthPosition) || 0));
  const face = faces[Math.abs(Number(ch.faceShape) || 0) % faces.length];
  const brow = ['M-12 0Q0-7 12-1','M-12-2 12-2','M-12 0Q0-5 12 0','M-12 3Q0-6 12-5','M-12-3Q0-11 12-3','M-12 1Q0-3 12 1'][Math.abs(Number(ch.eyebrow) || 0)%6];
  return <svg xmlns="http://www.w3.org/2000/svg" className="classio-avatar" data-animate={animate ? 'true' : 'false'} width={size} height={size} viewBox="0 0 200 220" role="img" aria-label={label || (ch.name ? `${ch.name}'s Classio character` : 'Classio character')}>
    <defs>
      <radialGradient id={id+'skin'} cx="35%" cy="28%" r="80%"><stop stopColor={shade(ch.skin,22)}/><stop offset=".62" stopColor={ch.skin}/><stop offset="1" stopColor={skinShadow}/></radialGradient>
      <linearGradient id={id+'hair'} x2=".8" y2="1"><stop stopColor={shade(ch.hair,16)}/><stop offset=".5" stopColor={ch.hair}/><stop offset="1" stopColor={hairShadow}/></linearGradient>
      <linearGradient id={id+'shirt'} x2="1" y2="1"><stop stopColor={shade(ch.top,18)}/><stop offset=".52" stopColor={ch.top}/><stop offset="1" stopColor={shade(ch.top,-25)}/></linearGradient>
      <radialGradient id={id+'back'} cx=".4" cy=".25" r=".9"><stop stopColor={shade(ch.bg,12)}/><stop offset="1" stopColor={ch.bg}/></radialGradient>
      <clipPath id={id+'clip'}><rect width="200" height="220" rx="34"/></clipPath>
      <clipPath id={id+'face'}><path d={face}/></clipPath>
    </defs>
    <g clipPath={`url(#${id}clip)`}>
      <rect width="200" height="220" fill={`url(#${id}back)`}/><ellipse cx="100" cy="218" rx="71" ry="12" fill="#142e38" opacity=".1"/>
      {[4,5,9,11,15].includes(hs) && <path d="M48 74C32 93 32 142 39 174L64 184 71 61h58l7 123 27-12c7-43 1-78-12-99Z" fill={`url(#${id}hair)`}/>}
      {hs===6 && <path d="M139 43c45 0 41 46 27 74l-18 37 4-57c6-23-1-31-16-33Z" fill={`url(#${id}hair)`}/>}
      {hs===7 && <><ellipse cx="100" cy="24" rx="26" ry="21" fill={`url(#${id}hair)`}/><path d="M83 19q13-12 29-5" fill="none" stroke={shade(ch.hair,34)} opacity=".32" strokeWidth="4" strokeLinecap="round"/></>}
      {hs===13 && [53,147].map(x=><circle key={x} cx={x} cy="36" r="21" fill={`url(#${id}hair)`}/>)}
      <path d="M23 222l7-29c4-19 25-30 47-33h46c22 3 43 14 47 33l7 29Z" fill={`url(#${id}shirt)`}/>
      <path d="M80 147v18c3 15 37 15 40 0v-18Z" fill={ch.skin}/><path d="M80 148q20 18 40 0v11q-20 16-40 0Z" fill={skinShadow} opacity=".55"/>
      <path d="M74 163q26 26 52 0" fill="none" stroke={shade(ch.top,-25)} strokeWidth="6"/>
      {[1,8].includes(top) && <><path d="M77 160q-37 4-29 35l33-18m42-17q37 4 29 35l-33-18" fill="none" stroke={shade(ch.top,27)} strokeWidth="10"/><path d="M83 178v27m34-27v27" stroke="#fff" opacity=".6" strokeWidth="2"/><path d="M80 213h40" stroke={shade(ch.top,-30)} strokeWidth="2"/></>}
      {[2,4,10,12,15].includes(top) && <><path d="m75 163 25 57 25-57-25 18Z" fill={top===4?'#f5f2ea':shade(ch.top,35)}/><path d="m75 163-10 21 19 1-8 13 24 22m25-57 10 21-19 1 8 13-24 22" fill="none" stroke={shade(ch.top,-28)} strokeWidth="2"/></>}
      {top===4 && <path d="m95 181 5-5 5 5-3 9 5 24h-14l5-24Z" fill="#965650"/>}
      {top===3 && <><path d="M30 197q5-22 27-27l-5 50H24Z" fill={ch.skin}/><path d="M170 197q-5-22-27-27l5 50h28Z" fill={ch.skin}/></>}
      {top===5 && <path d="M46 211h108v9H46Z" fill={ch.skin}/>}
      {top===6 && <><path d="m77 163 23 18-17 8-12-22m52-4-23 18 17 8 12-22" fill={shade(ch.top,34)}/>{[191,202,213].map(y=><circle key={y} cx="100" cy={y} r="1.8" fill={shade(ch.top,-40)}/>)}</>}
      {top===7 && <><path d="M39 179v41m122-41v41" stroke={shade(ch.top,80)} strokeWidth="6"/><text x="100" y="214" textAnchor="middle" fontFamily="system-ui" fontSize="27" fontWeight="800" fill={shade(ch.top,100)}>10</text></>}
      {top===9 && <path d="m68 165 32 29 32-29-8 55H76Z" fill={shade(ch.top,16)}/>}
      {top===11 && <path d="m100 186 5 9 10 2-8 7 1 10-8-5-9 5 2-10-8-7 10-2Z" fill="#fff" opacity=".8"/>}
      {top===12 && <text x="128" y="207" fontFamily="system-ui" fontSize="18" fontWeight="800" fill="#fff" opacity=".85">C</text>}
      {top===13 && <><path d="m76 164 48 56H93l-28-45m59-11-45 56" fill="none" stroke={shade(ch.top,50)} strokeWidth="6"/><path d="M57 211h86" stroke={shade(ch.top,-30)} strokeWidth="9"/></>}
      {top===14 && <path d="M78 157q22 10 44 0v16q-22 13-44 0Z" fill={`url(#${id}shirt)`}/>}
      {top===15 && <><path d="M63 191h19v15H63Zm55 0h19v15h-19ZM100 181v39" stroke={shade(ch.top,85)} fill="none" strokeWidth="1.5" strokeDasharray="3 2"/>{[190,201,212].map(y=><circle key={y} cx="100" cy={y} r="1.8" fill="#ccb77d"/>)}</>}
      {Number(ch.necklace)>0 && <><path d={`M76 167Q100 ${Number(ch.necklace)===4?184:200} 124 167`} fill="none" stroke={ch.necklaceColor} strokeWidth={Number(ch.necklace)===4?4:2.5}/>{[2,3,5].includes(Number(ch.necklace)) && <circle cx="100" cy="184" r={Number(ch.necklace)===2?5:3} fill={ch.necklaceColor}/>}</>}
      {[48,152].map(x=><g key={x}><ellipse cx={x} cy="94" rx="10" ry="17" fill={ch.skin}/><path d={`M${x+(x<100?1:-1)} 85q${x<100?-8:8} 7 0 15`} fill="none" stroke={skinShadow} strokeWidth="2.5" strokeLinecap="round"/></g>)}
      <g className="avatar-face">
        <path d={face} fill={`url(#${id}skin)`}/><ellipse cx="82" cy="49" rx="21" ry="10" fill="#fff" opacity=".12" clipPath={`url(#${id}face)`}/>
        {ch.blush && [68,132].map(x=><ellipse key={x} cx={x} cy="114" rx="12" ry="7" fill="#d7736d" opacity=".3"/>)}
        {ch.freckles && [0,1,2,3,4,5].map(index=><circle key={index} cx={index<3?62+index*6:124+(index-3)*6} cy={111+index%2*3} r="1.2" fill={shade(ch.skin,-60)} opacity=".65"/>)}
        <g className="avatar-eyes">
          {[-1,1].map(side=><g key={side} transform={`translate(${100+side*(23+eyeSpacing)} 91) scale(${featureScale('eyeSize')})`}>
            {classic && eye!==3 ? <>
              {eye===4 && <ellipse rx="8" ry="10" fill="#fffdf8"/>}
              <ellipse cx="0" cy="0" rx={eye===1?5.5:eye===2?6.5:4.5} ry={eye===1?3.8:eye===2?5:8} fill={eye===4?ch.eyes:shade(ch.eyes,-35)} transform={eye===2?'rotate(-12)':''}/>
              <ellipse cx="-1" cy="-2" rx="1" ry="1.5" fill="#fff" opacity=".5"/>
            </> : eye===3 ? <path d="M-10 2q10-13 20 0" fill="none" stroke="#2e3032" strokeWidth="3" strokeLinecap="round"/> : <>
              {eye!==1 && <ellipse cx="0" cy="0" rx={eye===2?12:10} ry={eye===2?7:eye===4?12:10} fill="#fffdf8"/>}
              {eye===1 && <path d="M-13 0Q0-14 13 0 0 11-13 0" fill="#fffdf8"/>}
              <ellipse cx="0" cy="1" rx={eye===2?5:5.8} ry={eye===2?6:7.5} fill={ch.eyes}/><ellipse cx="0" cy="2" rx="3" ry="4.5" fill="#202829"/>
              <circle cx="-2" cy="-2" r="2" fill="#fff"/><circle cx="2" cy="4" r=".85" fill="#fff" opacity=".7"/>
              <path d="M-11-3q10-12 22 0" fill="none" stroke={shade(ch.skin,-65)} strokeWidth="1.7" strokeLinecap="round" opacity=".8"/>
            </>}
          </g>)}
        </g>
        {[-1,1].map(side=><path key={side} d={brow} transform={`translate(${100+side*(23+eyeSpacing)} ${72+Math.max(-6,Math.min(6,Number(ch.browHeight)||0))}) scale(${side} 1)`} fill="none" stroke={ch.hair} strokeWidth={Number(ch.eyebrow)===2?5.5:3.5} strokeLinecap="round"/>)}
        <g transform={`translate(100 108) scale(${featureScale('noseSize')}) translate(-100 -108)`}>
        {classic && Number(ch.nose)===0 ? <><path d="m100 97-8 17 15 1Z" fill={skinShadow} opacity=".65"/><path d="m100 97 6 15-10-2Z" fill={shade(ch.skin,20)}/></> : Number(ch.nose)===1 ? <path d="M98 96q-5 14-4 17 5 5 13 0" fill="none" stroke={skinShadow} strokeWidth="2.4" strokeLinecap="round"/> : Number(ch.nose)===2 ? <path d="m100 96-5 16 10 1" fill="none" stroke={skinShadow} strokeWidth="2.6" strokeLinecap="round"/> : <><ellipse cx="100" cy="111" rx={Number(ch.nose)===3?10:7} ry="5" fill={skinShadow} opacity=".65"/><ellipse cx="98" cy="108" rx="5" ry="3" fill={shade(ch.skin,16)} opacity=".9"/></>}
        </g>
        {facial>0 && facial!==2 && <path d={facial===1?'M63 120Q67 154 100 158q33-4 37-38l-7 26q-30 27-60 0Z':facial===5?'M86 139q14 11 28 0l-5 19H91Z':facial===3?'M61 116q12 27 39 28t39-28l-8 29q-31 28-62 0Z':'M59 111q12 32 41 33t41-33l-6 37q-35 33-70 0Z'} fill={ch.hair} opacity={facial===1?.35:1}/>}
        <g transform={`translate(100 ${132+mouthY}) scale(${featureScale('mouthSize')*(classic ? .72 : 1)}) translate(-100 -132)`}>
          {[2,5].includes(Number(ch.mouth)) ? <><path d="M82 128q18 5 36 0c-2 20-34 20-36 0" fill="#754139"/><path d="M84 129q16 4 32 0l-3 7H87Z" fill="#fffdf8"/></> : <path d={Number(ch.mouth)===1?'M85 136q15-13 30 0':Number(ch.mouth)===3?'M88 132q17 6 29-4':Number(ch.mouth)===4?'M88 133h24':Number(ch.mouth)===6?'M89 130q11 9 22 0':'M84 129q16 15 32 0'} fill="none" stroke={ch.lips?ch.lipColor:'#9a5c50'} strokeWidth={ch.lips?4:2.8} strokeLinecap="round"/>}
          {ch.lips && <path d="M94 136h10" stroke="#fff" strokeWidth="1" opacity=".45"/>}
        </g>
        {[2,4].includes(facial) && <path d="M100 119q-11-7-17 5 10 4 17-1 7 5 17 1-6-12-17-5" fill={ch.hair}/>}
      </g>
      <path d={hairFront[hs]} fill={`url(#${id}hair)`}/>
      {[9,11,15].includes(hs) && [52,61,139,148].map((x,index)=><g key={x}><path d={`M${x} 55q${index<2?-6:6} 35 ${index<2?-4:4} 87`} stroke={ch.hair} strokeWidth={hs===9?7:9} fill="none" strokeLinecap="round"/><path d={`M${x-1} 58q${index<2?-6:6} 30 ${index<2?-4:4} 76`} stroke={shade(ch.hair,30)} strokeWidth="1.4" fill="none" opacity=".4"/></g>)}
      {[8,10].includes(hs) && Array.from({length:11},(_,index)=><circle key={index} cx={54+index*9.2} cy={32-Math.sin(index/10*Math.PI)*12} r={hs===8?13:9} fill={index%3===0?shade(ch.hair,10):ch.hair}/>)}
      {![8,10,12,14].includes(hs) && <path d="M65 35q26-17 53-7" fill="none" stroke={shade(ch.hair,48)} strokeWidth="3" strokeLinecap="round" opacity=".2"/>}
      {glasses>0 && <g stroke={ch.glassesColor} strokeWidth="3" fill={glasses===4?ch.glassesColor:'none'}>
        {[-1,1].filter(side=>glasses!==7||side===1).map(side=><g key={side} transform={`translate(${100+side*(23+eyeSpacing)} 93)`}>{[1,7].includes(glasses)?<circle r="16"/>:glasses===5?<path d="M0-7c-15-19-30 7 0 23 30-16 15-42 0-23Z"/>:glasses===3?<path d="m-17-13 31 3c4 29-36 29-31-3Z"/>:glasses===6?<path d="M-16-7q16-10 32 0l-3 18q-13 14-26 0Z"/>:<rect x="-17" y="-13" width="34" height="27" rx="7"/>}</g>)}
        {glasses!==7 && <path d={`M${94-eyeSpacing} 92q6-5 ${12+eyeSpacing*2} 0M44 87l10 3m92 0 10-3`} fill="none"/>}
      </g>}
      {Number(ch.earring)>0 && [48,152].map(x=><g key={x} fill={ch.earringColor} stroke={ch.earringColor} strokeWidth="2">{Number(ch.earring)===3?<circle cx={x} cy="113" r="6" fill="none"/>:Number(ch.earring)===2||Number(ch.earring)===5?<><circle cx={x} cy="109" r="2"/><path d={`m${x} 112-4 10 4 5 4-5Z`}/></>:Number(ch.earring)===6?<path d={`M${x} 109v15m-5-10h10`}/>:Number(ch.earring)===4?<path d={`m${x} 106 2 4 5 1-4 3 1 5-4-2-4 2 1-5-4-3 5-1Z`}/>:<circle cx={x} cy="109" r="3"/>}</g>)}
      {hat===1 && <><path d="M45 52C44 11 157 11 155 52Z" fill={ch.hatColor}/><rect x="44" y="43" width="112" height="17" rx="6" fill={shade(ch.hatColor,17)}/><path d="M60 46v11m14-11v11m14-11v11m14-11v11m14-11v11m14-11v11m14-11v11" stroke={shade(ch.hatColor,-15)} opacity=".5"/></>}
      {hat===2 && <><path d="M46 51C46 8 153 8 154 51Z" fill={ch.hatColor}/><path d="M91 48q56-14 77 9-42 12-77-9" fill={shade(ch.hatColor,-17)}/><path d="M100 23v22" stroke="#fff" opacity=".2"/></>}
      {hat===3 && <><path d="m64 40 3-24q14-6 33 5 19-11 33-5l3 24Z" fill={ch.hatColor}/><path d="M29 36q13 30 71 11 58 19 71-11-4 38-71 24-67 14-71-24" fill={shade(ch.hatColor,15)}/></>}
      {hat===4 && <><path d="m35 29 65-23 65 23-65 24Z" fill={ch.hatColor}/><path d="M63 42v16q37 9 74 0V42" fill={shade(ch.hatColor,-18)}/><path d="m151 34 8 38" stroke="#d7b66a" strokeWidth="3"/></>}
      {hat===5 && <path d="m56 48-6-32 24 18 26-27 26 27 24-18-6 32Z" fill={ch.hatColor} stroke={shade(ch.hatColor,-28)} strokeWidth="2"/>}
      {hat===6 && <><path d="M70 45 100 0l30 45Z" fill={ch.hatColor}/><circle cx="100" cy="5" r="6" fill="#f8f2da"/></>}
      {hat===7 && <path d="M49 48q51-38 102 0" fill="none" stroke={ch.hatColor} strokeWidth="10"/>}
      {hat===8 && <><path d="m64 16 72 0 10 35H54Z" fill={ch.hatColor}/><path d="M54 47q46 12 92 0l13 16q-59 18-118 0Z" fill={shade(ch.hatColor,12)}/></>}
    </g>
  </svg>;
}
export default memo(Avatar);

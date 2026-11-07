import { useId } from "react";

const color = (value, fallback) => /^#[\da-f]{3}(?:[\da-f]{3})?$/i.test(value || "") ? value : fallback;
const tint = (hex, light) => {
  const raw = hex.slice(1), full = raw.length === 3 ? [...raw].map(value => value + value).join("") : raw;
  return "#" + [0, 2, 4].map(index => { const value = parseInt(full.slice(index, index + 2), 16); return Math.round(value + (light > 0 ? 255 - value : value) * light).toString(16).padStart(2, "0"); }).join("");
};
const cap = "M18 42C14 17 28 7 50 7C73 7 87 20 82 43L76 39C73 26 69 24 64 23C51 33 38 28 29 34L24 43Z";
const fringes = [
  "M20 39C17 19 31 10 50 10C69 10 83 20 80 39L75 31L25 31Z",
  "M18 40C15 18 31 7 52 8C71 8 86 22 81 41L73 29C66 24 61 24 57 25L33 37L30 28L24 40Z",
  "M18 43C13 18 29 7 50 7C72 7 87 21 82 43L74 30L67 39L57 28L48 38L39 28L29 40L25 30L23 43Z",
  "M18 42C9 26 21 5 36 9C42 1 56 5 61 8C78 6 89 22 82 43L73 31L61 35L51 27L39 35L28 30L24 42Z",
  cap, cap, cap, cap,
  "M17 44C7 40 9 27 13 22C10 12 24 7 30 8C35 0 48 3 52 5C62 0 75 7 77 12C91 11 94 27 87 33C92 42 83 46 79 44L74 31L63 30L53 26L42 31L30 29L24 43Z",
  cap, "M18 43C10 31 16 10 31 10C38 1 51 3 57 8C74 1 90 22 82 44L75 32L65 35L54 29L44 35L31 30L23 42Z",
  cap, "M19 41C14 23 27 14 39 13L47 1L57 9L68 6L66 16C81 21 85 31 80 42L73 31L60 28L49 32L33 28L24 41Z",
  cap, "M20 42C17 23 28 15 41 14L47 1L54 11L61 5L60 17C78 17 85 29 80 42L73 32L26 32Z", cap,
  "M18 41C16 18 32 8 53 8C73 8 86 23 81 41L74 29C58 25 41 29 26 29L24 42Z",
];

export default function FriendlyAvatar({ character = {}, size = 40 }) {
  const id = useId().replace(/[^\w-]/g, "");
  const ch = character;
  const skin = color(ch.skin, "#f5c89a"), hair = color(ch.hair, "#493326"), shirt = color(ch.top, "#417f99");
  const eye = color(ch.eyes, "#373b38"), hat = color(ch.hatColor, "#5e91aa"), frame = color(ch.glassesColor, "#343c43");
  const style = Number(ch.hairStyle) || 0;
  const long = [4, 5, 9, 11, 15].includes(style);
  const mouth = Number(ch.mouth) || 0;
  const brows = Number(ch.eyebrow) || 0;
  const top = Number(ch.topStyle) || 0;
  const jewelry = color(ch.necklaceColor, "#c99d48");
  const expression = mouth === 1 ? "M43 68Q50 62 57 68" : mouth === 3 ? "M43 65Q50 70 58 63" : mouth === 4 ? "M44 66h12" : "M43 64Q50 72 57 64";
  const hairFill = `url(#hair-${id})`, skinFill = `url(#skin-${id})`, shirtFill = `url(#shirt-${id})`;
  const lips = ch.lips ? color(ch.lipColor, "#b96565") : "#6f3f38";

  return <svg width={size} height={size} viewBox="0 0 100 100" role="img" aria-label="Your character" className="friendly-avatar">
    <defs>
      <radialGradient id={`skin-${id}`} cx="32%" cy="23%" r="88%"><stop stopColor={tint(skin, .16)} /><stop offset=".65" stopColor={skin} /><stop offset="1" stopColor={tint(skin, -.18)} /></radialGradient>
      <linearGradient id={`hair-${id}`} x1="20%" y1="0" x2="75%" y2="100%"><stop stopColor={tint(hair, .18)} /><stop offset=".45" stopColor={hair} /><stop offset="1" stopColor={tint(hair, -.22)} /></linearGradient>
      <linearGradient id={`shirt-${id}`} x2="0" y2="1"><stop stopColor={tint(shirt, .15)} /><stop offset="1" stopColor={tint(shirt, -.16)} /></linearGradient>
      <clipPath id={`avatar-${id}`}><rect width="100" height="100" rx="24" /></clipPath></defs>
    <g clipPath={`url(#avatar-${id})`}>
      <rect width="100" height="100" fill={color(ch.bg, "#e5f0f7")} />
      {/* Every extension overlaps the crown, keeping the silhouette connected. */}
      {long && <path d={style === 9 ? "M18 31C15 16 32 8 50 8C72 8 85 22 82 40L84 77Q78 87 70 76L66 48H34L30 76Q22 87 16 77Z" : "M18 31C15 16 32 8 50 8C72 8 85 22 82 40L87 84Q75 91 65 83H34Q21 89 13 82Z"} fill={hairFill} />}
      {style === 6 && <path d="M71 19C91 12 98 28 94 44L87 70L78 65L81 37L68 31Z" fill={hairFill} />}
      {style === 7 && <circle cx="58" cy="12" r="13" fill={hairFill} />}
      {style === 13 && <g fill={hairFill}><circle cx="26" cy="20" r="13" /><circle cx="74" cy="20" r="13" /></g>}
      <path d="M18 104L21 88Q24 79 39 78H61Q76 79 79 88L84 104Z" fill={shirtFill} />
      {top === 1 && <path d="M35 82Q50 97 65 82M37 84v15M63 84v15" stroke="#fff" strokeOpacity=".65" strokeWidth="2" fill="none" />}
      {[2, 4, 10, 12, 15].includes(top) && <path d="m33 81 11 14 6-10 6 10 11-14M50 89v16" stroke="#fff" strokeOpacity=".6" strokeWidth="2" fill="none" />}
      {[3, 5, 9].includes(top) && <path d="M31 82v19M69 82v19" stroke={skin} strokeWidth="8" />}
      {top === 6 && <path d="m36 80 8 12 6-7 6 7 8-12" fill="#fff" fillOpacity=".7" />}
      {[7, 8, 11].includes(top) && <path d="M24 92h52M23 97h54" stroke="#fff" strokeOpacity={top === 11 ? ".65" : ".35"} strokeWidth="3" />}
      {top === 13 && <path d="m33 80 24 23M67 80l-24 23" stroke="#fff" strokeOpacity=".6" strokeWidth="3" />}
      {top === 14 && <path d="M38 77h24v12H38z" fill={shirtFill} stroke="#fff" strokeOpacity=".2" />}
      {top === 0 && <path d="M38 82q12 12 24 0" stroke="#fff" strokeOpacity=".6" strokeWidth="2" fill="none" />}
      {top === 5 && <path d="m46 95 4-3 5 4-5 5Z" fill="#fff" fillOpacity=".65" />}
      {top === 7 && <text x="50" y="98" textAnchor="middle" fontSize="12" fontWeight="bold" fill="#fff">1</text>}
      {top === 9 && <><path d="m37 80 13 8 13-8" fill="#fff" fillOpacity=".5" /><circle cx="50" cy="94" r="1.5" fill="#fff" /></>}
      {top === 11 && <path d="m50 87 6 6-6 6-6-6Z" fill="#fff" fillOpacity=".8" />}
      {top === 12 && <><path d="M24 85v16M76 85v16" stroke="#fff" strokeOpacity=".65" strokeWidth="7" /><text x="61" y="97" fontSize="9" fill="#fff" fontWeight="bold">C</text></>}
      {top === 15 && <><path d="M30 89h12v9H30zm28 0h12v9H58z" fill="none" stroke="#fff" strokeOpacity=".4" /><circle cx="50" cy="96" r="1" fill="#e5c477" /></>}
      <path d="M43 72h14v13q-7 8-14 0Z" fill={skinFill} /><path d="M43 76q7 7 14 0v4q-7 7-14 0Z" fill="#603a22" opacity=".1" />
      <g fill={skinFill}><ellipse cx="20" cy="49" rx="8" ry="10" /><ellipse cx="80" cy="49" rx="8" ry="10" />
        <path d="M21 36C21 16 35 12 50 12C68 12 80 24 79 43L77 58C75 72 63 81 50 81C36 81 24 72 22 58Z" /></g>
      <path d="M16 47q6-4 7 7M84 47q-6-4-7 7" fill="none" stroke="#9e624b" strokeOpacity=".35" strokeWidth="2" />
      <path d="M26 59q5 16 24 17q20-1 24-17" fill="none" stroke={tint(skin, -.25)} strokeOpacity=".3" strokeWidth="1.4" />
      <g fill={hairFill} data-hair-style={style}>
        <path d={fringes[style] || cap} data-hair-crown="connected" />
        {style === 9 && <g stroke="#fff" strokeOpacity=".2" strokeWidth="1.5" fill="none"><path d="m20 32 6 6-7 6 7 6-7 6 7 6-7 6 7 6M80 32l-6 6 7 6-7 6 7 6-7 6 7 6-7 6" /></g>}
        {style === 11 && <g stroke={hair} strokeWidth="7" strokeLinecap="round"><path d="M24 26 17 77M31 24 25 82M76 25l8 53M69 25l7 57" /><path d="M33 20 38 30M42 17l6 13M53 17l7 13M65 20l5 10" stroke="#fff" strokeOpacity=".13" strokeWidth="2" /></g>}
        {style === 15 && <g stroke={hair} strokeWidth="8" strokeLinecap="round"><path d="M25 25 16 69M32 23 24 80M75 25l10 44M68 23l8 57M34 21l3 17M46 18l3 19M58 20l3 17M69 24l3 12" /><path d="m18 53 6 2m53-2 6-2" stroke="#c9a35b" strokeWidth="3" /></g>}
        {style === 5 && <path d="M18 33Q9 42 18 52Q12 66 20 77L28 73Q20 62 25 53L24 33M82 33Q91 43 82 52Q89 67 80 77L73 73Q80 60 75 52L77 33Z" />}
      </g>
      <path d="M27 24q9-12 25-11" stroke="#fff" strokeOpacity=".12" strokeWidth="3" strokeLinecap="round" fill="none" />
      <g fill="none" stroke={hair} strokeWidth={brows === 2 ? 3.2 : 2.2} strokeLinecap="round">
        <path d={brows === 4 ? "M30 36q5-7 10-3M60 35q5-2 10 1" : brows === 5 ? "m30 37 10-1m20 0 10 1" : brows === 1 ? "M30 36h10M60 36h10" : brows === 3 ? "m30 37 10-4m20 0 10 4" : "M30 36q5-4 10 0M60 36q5-4 10 0"} />
      </g>
      {ch.eyeShape === 4 ? <g fill="none" stroke={eye} strokeWidth="3" strokeLinecap="round"><path d="M29 48q6-10 12 0M59 48q6-10 12 0" /></g> : <>
        <g fill="#fff"><ellipse cx="35" cy="47" rx={ch.eyeShape === 5 ? 8 : 7} ry={ch.eyeShape === 1 ? 6 : 8} /><ellipse cx="65" cy="47" rx={ch.eyeShape === 5 ? 8 : 7} ry={ch.eyeShape === 1 ? 6 : 8} /></g>
        <g fill={eye}><ellipse cx="36" cy="48" rx="4.8" ry={ch.eyeShape === 1 ? 5.5 : 6.5} /><ellipse cx="64" cy="48" rx="4.8" ry={ch.eyeShape === 1 ? 5.5 : 6.5} /></g>
        <g fill="#1d282c"><ellipse cx="36" cy="48" rx="2.6" ry="4.8" /><ellipse cx="64" cy="48" rx="2.6" ry="4.8" /></g>
        <g fill="#fff"><circle cx="34.5" cy="45.5" r="1.8" /><circle cx="62.5" cy="45.5" r="1.8" /><circle cx="38" cy="51" r=".8" /><circle cx="66" cy="51" r=".8" /></g>
        {ch.eyeShape === 3 && <path d="M27 42h16v5H27zm30 0h16v5H57z" fill={skinFill} />}
      </>}
      {ch.eyeShape === 2 && <path d="m27 42-3-2m49 2 3-2m-47 6-4-1m50 1 4-1" stroke={hair} strokeWidth="2" />}
      <path d="m50 52-4 7q4 3 8 0Z" fill="#b97759" fillOpacity=".65" />
      {ch.blush && <g fill="#dc7e80" opacity=".45"><ellipse cx="30" cy="60" rx="6" ry="3" /><ellipse cx="70" cy="60" rx="6" ry="3" /></g>}
      {ch.freckles && <g fill="#985f44">{[28, 32, 36, 64, 68, 72].map((x, i) => <circle key={x} cx={x} cy={59 + i % 2 * 2} r=".9" />)}</g>}
      {ch.facialHair > 0 && <g fill={hairFill}>
        {[1, 3, 4, 5].includes(ch.facialHair) && <path d="M29 65q21 24 42 0l-4 10q-17 17-34 0Z" opacity={ch.facialHair === 1 ? ".25" : ".9"} />}
        {[2, 3, 4].includes(ch.facialHair) && <path d="M50 60q-8-4-13 6q8 3 13-1q5 4 13 1q-5-10-13-6Z" />}
      </g>}
      {mouth >= 5 ? <g><path d={mouth === 5 ? "M40 63q10 17 20 0Z" : "M41 63q9-3 18 0q-1 15-9 13q-8-1-9-13Z"} fill="#733f3a" /><path d="M43 64h14l-2 4H45Z" fill="#fff" /><path d="M45 72q5-3 10 0" stroke="#dc9892" strokeWidth="3" /></g> : mouth === 2 ? <ellipse cx="50" cy="66" rx="4.5" ry="5" fill={lips} /> : <path d={expression} fill="none" stroke={lips} strokeWidth={ch.lips ? 3 : 2.2} strokeLinecap="round" />}
      {ch.glasses > 0 && <g stroke={frame} strokeWidth="2.1" fill={ch.glasses === 4 ? "#243441" : "none"}>
        {ch.glasses === 2 || ch.glasses === 4 ? <><rect x="25" y="40" width="20" height="15" rx={ch.glasses === 4 ? 6 : 3} /><rect x="55" y="40" width="20" height="15" rx={ch.glasses === 4 ? 6 : 3} /></>
          : ch.glasses === 3 ? <path d="M22 39q23-3 23 8q-1 13-16 6ZM78 39q-23-3-23 8q1 13 16 6Z" />
          : ch.glasses === 5 ? <path d="M35 55 24 45C19 35 33 33 35 40C40 31 50 39 45 46ZM65 55 54 45C49 35 63 33 65 40C70 31 80 39 75 46Z" />
          : ch.glasses === 6 ? <path d="M24 42q9-5 21 0q2 17-11 14q-10-2-10-14ZM56 42q9-5 21 0q2 17-11 14q-10-2-10-14Z" />
          : <><ellipse cx="35" cy="47" rx="11" ry="10" />{ch.glasses !== 7 && <ellipse cx="65" cy="47" rx="11" ry="10" />}</>}
        {ch.glasses !== 7 ? <path d="M46 45q4-3 8 0M24 44h-5M76 44h5" fill="none" /> : <path d="M25 54q-8 17 4 28" strokeWidth="1" />}
      </g>}
      {ch.necklace > 0 && <g stroke={jewelry} strokeWidth={ch.necklace === 4 ? 4 : 1.8} fill="none"><path d={ch.necklace === 4 ? "M39 82q11 7 22 0" : "M36 84q14 15 28 0"} />
        {ch.necklace === 2 && <path d="m50 91 3 5-3 4-3-4Z" fill={jewelry} />}
        {[3, 5].includes(ch.necklace) && [38, 43, 50, 57, 62].map((x, i) => <circle key={x} cx={x} cy={87 + (2 - Math.abs(i - 2)) * 2} r={ch.necklace === 5 ? 2.4 : 2} fill={ch.necklace === 5 ? "#f5f0e8" : jewelry} />)}
      </g>}
      {ch.earring > 0 && <g fill={color(ch.earringColor, "#c99d48")} stroke={color(ch.earringColor, "#c99d48")}>
        {[18, 82].map(x => <g key={x}>{ch.earring === 3 ? <ellipse cx={x} cy="60" rx="3" ry="4" fill="none" strokeWidth="1.5" />
          : ch.earring === 4 ? <path d={`m${x} 55 1.5 3 3 .5-2.2 2.2.5 3-2.8-1.5-2.8 1.5.5-3-2.2-2.2 3-.5Z`} />
          : ch.earring === 6 ? <path d={`M${x} 56v10m-3-6h6`} fill="none" strokeWidth="2" />
          : <><circle cx={x} cy="57" r="1.8" />{[2, 5].includes(ch.earring) && <><path d={`M${x} 58v5`} /><ellipse cx={x} cy="65" rx={ch.earring === 2 ? 2.3 : 1.5} ry={ch.earring === 2 ? 3.4 : 2} /></>}</>}</g>)}
      </g>}
      {ch.hat > 0 && <g fill={hat} stroke={hat} strokeLinejoin="round">
        {ch.hat === 1 && <><path d="M21 27C22 0 78 0 79 27Z" /><rect x="19" y="24" width="62" height="9" rx="3" /></>}
        {ch.hat === 2 && <><path d="M22 28C25 3 73 3 78 28Z" /><path d="M38 28q29-4 50 8q-19 6-39-1Z" /></>}
        {ch.hat === 3 && <path d="M9 28q9 6 16-2l7-17q18 7 36 0l7 17q7 8 16 2q-6 16-41 13Q15 44 9 28Z" />}
        {ch.hat === 4 && <><path d="m13 18 37-13 37 13-37 13Z" /><path d="M29 23v12q21 8 42 0V23M82 20v19" strokeWidth="3" /></>}
        {ch.hat === 5 && <path d="M24 31 18 10l17 10L50 4l15 16 17-10-6 21Z" />}
        {ch.hat === 6 && <path d="M32 29 50 0l18 29Z" />}
        {ch.hat === 7 && <path d="M21 32q29-18 58 0" fill="none" strokeWidth="7" />}
        {ch.hat === 8 && <path d="m29 10 42 0 5 19 10 7q-36 11-72 0l10-7Z" />}
      </g>}
    </g>
  </svg>;
}

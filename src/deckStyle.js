import { contrastRatio } from './styleConfig';

const color = (value, fallback) => /^#[\da-f]{6}$/i.test(value || '') ? value : fallback;
const clamp = (value, min, max, fallback) => Number.isFinite(Number(value)) ? Math.max(min, Math.min(max, Number(value))) : fallback;
const rgb = hex => hex.slice(1).match(/../g).map(n => parseInt(n, 16));
function composite(foreground, background, opacity) {
  const front = rgb(foreground), back = rgb(background);
  return '#' + front.map((channel, i) => Math.round(channel * opacity + back[i] * (1 - opacity)).toString(16).padStart(2, '0')).join('');
}

export function normalizeDeckDesign(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('The AI did not return a presentation design. Your existing slides are kept.');
  }
  const layout = ['lecture', 'editorial', 'visual'].includes(value.layout) ? value.layout : 'visual';
  const bg = color(value.bg, '#f7f9fc'), requestedAccent = color(value.accent, '#286a9b');
  const pattern = ['none', 'checkerboard', 'grid', 'dots', 'stripes', 'ruled'].includes(value.pattern) ? value.pattern : 'none';
  const patternColor = color(value.patternColor, requestedAccent), patternOpacity = clamp(value.patternOpacity, .04, .16, .09);
  // A readable base color can become illegible over a dark checker or stripe.
  const backgrounds = pattern === 'none' ? [bg] : [bg, composite(patternColor, bg, patternOpacity)];
  const contrast = ink => Math.min(...backgrounds.map(background => contrastRatio(ink, background)));
  const fallback = ['#ffffff', '#102332'].sort((a, b) => contrast(b) - contrast(a))[0];
  const readable = candidate => contrast(candidate) >= 4.5 ? candidate : fallback;
  const text = readable(color(value.text, '#253648'));
  return {
    name: String(value.name || value.label || 'Custom presentation').replace(/[<>]/g, '').slice(0, 60),
    description: String(value.description || 'A design based on your instructions.').replace(/[<>]/g, '').slice(0, 220),
    layout, bg, accent: readable(requestedAccent), text, sub: readable(color(value.sub, text)),
    font: ['sans', 'serif', 'handwritten'].includes(value.font) ? value.font : layout === 'editorial' ? 'serif' : 'sans',
    pattern, patternColor, patternOpacity, radius: clamp(value.radius, 0, 20, 10),
  };
}

export function designFromInstructions(value, prompt = '') {
  const next = normalizeDeckDesign(value), text = prompt.toLowerCase();
  const noChecker = /\b(?:no|not|without|remove)\b.{0,16}\b(?:checkerboard|checker board|checkered|chequered)\b/.test(text);
  if (noChecker && next.pattern === 'checkerboard') next.pattern = 'none';
  else if (!noChecker && /\b(?:checkerboard|checker board|checkered|chequered)\b/.test(text)) next.pattern = 'checkerboard';
  if (/\bhandwrit(?:ten|ing)\b/.test(text)) next.font = 'handwritten';
  return normalizeDeckDesign(next);
}

export const DECK_STYLE_GUIDE = `Each design must be an object with name, description, layout (lecture|editorial|visual), font (sans|serif|handwritten), bg, accent, text, sub, pattern (none|checkerboard|grid|dots|stripes|ruled), patternColor, patternOpacity (0.04 to 0.16), radius (0 to 20). All colors are six-digit HEX. Follow the learner's style instructions, including actual background patterns and typography. Keep body text readable. Do not return CSS, HTML, code, URLs or image assets. When asked for several options, vary the composition, typography, pattern and palette meaningfully; do not repeat three predefined templates.`;

export function deckStyleVariables(design, theme) {
  if (!design) return {};
  const d = normalizeDeckDesign({ ...design, bg: theme.bg, text: theme.text, sub: theme.sub, accent: theme.accent });
  const ink = `rgba(${rgb(d.patternColor).join(',')},${d.patternOpacity})`;
  const font = d.font === 'serif' ? '"Fraunces Variable",Georgia,serif' : d.font === 'handwritten' ? '"Caveat Variable",cursive' : '"DM Sans Variable",Arial,sans-serif';
  const patterns = {
    checkerboard: `conic-gradient(${ink} 25%,transparent 0 50%,${ink} 0 75%,transparent 0)`,
    grid: `linear-gradient(${ink} 1px,transparent 1px),linear-gradient(90deg,${ink} 1px,transparent 1px)`,
    dots: `radial-gradient(${ink} 1.5px,transparent 1.5px)`,
    stripes: `repeating-linear-gradient(135deg,${ink} 0 10px,transparent 10px 24px)`,
    ruled: `linear-gradient(transparent calc(100% - 1px),${ink} 1px)`,
  };
  return {
    '--deck-font': font, '--deck-radius': d.radius + 'px',
    '--slide-accent': d.accent, '--slide-text': d.text, '--slide-sub': d.sub,
    backgroundImage: patterns[d.pattern] || 'none',
    backgroundSize: d.pattern === 'checkerboard' ? '12cqw 12cqw' : d.pattern === 'ruled' ? '100% 5cqw' : d.pattern === 'stripes' ? 'auto' : '4cqw 4cqw',
  };
}

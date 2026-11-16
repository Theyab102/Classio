// Repair common model JSON string escaping without evaluating generated code.
const commands = /^(?:frac|dfrac|tfrac|sqrt|text|textrm|textbf|mathrm|mathbf|mathbb|mathcal|operatorname|begin|end|left|right|times|cdot|div|pm|mp|neq|leq|geq|approx|equiv|infty|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|rho|sigma|tau|phi|omega|Delta|Sigma|Omega|sum|prod|int|sin|cos|tan|log|ln|lim|vec|overline|underline|hat|dot|ddot|nabla|boxed|quad|qquad|qquad|rightarrow|to|ce|pu)\b/;
export function parseAiJson(raw) {
  const source = String(raw || '').trim().replace(/^```(?:json)?\s*|\s*```$/gi, '');
  if (source.length > 250000) throw new Error('The generated response was too large. Try a smaller request.');
  let repaired = '', inside = false;
  for (let i = 0; i < source.length; i++) {
    const char = source[i];
    if (char === '"') { inside = !inside; repaired += char; continue; }
    if (inside && char === '\\') {
      const next = source[i + 1];
      if (next === '\\' || next === '"' || next === '/') { repaired += char + next; i++; }
      else if (commands.test(source.slice(i + 1)) || !next || !/[bfnrtu]/.test(next) || (next === 'u' && !/^[\da-f]{4}/i.test(source.slice(i + 2, i + 6)))) repaired += '\\\\';
      else { repaired += char + next; i++; }
    } else if (inside && char.charCodeAt(0) < 32) repaired += JSON.stringify(char).slice(1, -1);
    else repaired += char;
  }
  try { return JSON.parse(repaired); }
  catch { throw new Error('The AI response was incomplete or incorrectly formatted. Your current work is kept. Try again or request fewer items.'); }
}

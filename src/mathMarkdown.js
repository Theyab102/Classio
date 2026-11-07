import { Marked } from "marked";
import katex from "katex";

export const MATH_OUTPUT_GUIDE = String.raw`Write mathematics as LaTeX, using \( ... \) for inline math and \[ ... \] for display equations. Keep units inside math, for example \(100\,\mathrm{m/s}\). Never put equations in code fences. When returning JSON, encode each LaTeX backslash as two backslashes, for example {"answer":"\\(F=ma\\)"}, so the decoded string retains its delimiters. Define symbols, show substitutions with units, and check arithmetic and dimensions in worked examples. Use Markdown headings, bold text, lists and GFM tables for other content.`;
const escape = text => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
export function normalizeLatex(value) {
  return value.replace(/(?<!\\)\\\\(?=(?:frac|sqrt|theta|text|mathrm|times|approx|cdot|sin|cos|tan|alpha|beta|gamma|Delta|pi|left|right|vec|sum|int|begin|end)\b)/g, "\\")
    .replaceAll(String.fromCharCode(12) + "rac", "\\frac").replaceAll(String.fromCharCode(8) + "egin", "\\begin")
    .replaceAll(String.fromCharCode(9) + "heta", "\\theta").replaceAll(String.fromCharCode(9) + "ext{", "\\text{").replaceAll(String.fromCharCode(9) + "imes", "\\times");
}
export function equationHtml(latex, display = false) {
  const value = normalizeLatex(latex).trim().slice(0, 5000);
  return `<span data-classio-equation="true" data-latex="${escape(value)}" data-display="${display ? "block" : "inline"}">${escape(`\\(${value}\\)`)}</span>`;
}

// Tokenize math before Markdown consumes backslash escapes. Code stays code.
const parser = new Marked({ gfm: true, extensions: [
  { name: "classioDisplayMath", level: "block", start: source => source.search(/(?:^|\n)[ \t]{0,3}(?:\\\[|\$\$)/),
    tokenizer(source) {
      const match = /^(?:[ \t]{0,3}\\\[([\s\S]+?)\\\]|[ \t]{0,3}\$\$([\s\S]+?)\$\$)[ \t]*(?:\n|$)/.exec(source);
      if (match) return { type: "classioDisplayMath", raw: match[0], latex: match[1] || match[2] };
    }, renderer: token => `<p>${equationHtml(token.latex, true)}</p>\n` },
  { name: "classioInlineMath", level: "inline", start: source => source.search(/\\[([]|\$/),
    tokenizer(source) {
      const match = /^(?:\\\(([\s\S]+?)\\\)|\\\[([\s\S]+?)\\\]|\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$(?!\d))/.exec(source);
      if (match) return { type: "classioInlineMath", raw: match[0], latex: match[1] || match[2] || match[3] || match[4], display: Boolean(match[2] || match[3]) };
    }, renderer: token => equationHtml(token.latex, token.display) },
] });

export function parseStudyMarkdown(source) {
  const text = String(source || "").replace(/^\s*```(?:markdown|md)\s*\n([\s\S]*)\n```\s*$/i, "$1");
  return parser.parse(text);
}

export function recoverLegacyMath(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT), nodes = [];
  while (walker.nextNode()) if (!walker.currentNode.parentElement.closest("pre,code,[data-classio-equation],math,.katex")) nodes.push(walker.currentNode);
  for (const node of nodes) {
    // Old saved HTML may have lost its delimiter slashes. Keep nested
    // parentheses in worked examples such as (500 = \tfrac{1}{2}(9.8)t^2).
    const value = node.textContent; let start = 0, found = false;
    const fragment = document.createDocumentFragment();
    for (let i = 0; i < value.length; i++) {
      if (value[i] !== "(" && value[i] !== "[") continue;
      const open = value[i], close = open === "(" ? ")" : "]";
      let depth = 1, end = i + 1;
      for (; end < value.length && end - i <= 5000; end++) {
        if (value[end] === open) depth++;
        else if (value[end] === close && --depth === 0) break;
      }
      if (depth) continue;
      const latex = value.slice(i + 1, end);
      if (!/\\[a-zA-Z]+|[A-Za-z][_^](?:\{[^}]+\}|[A-Za-z0-9])/.test(latex)) continue;
      found = true; fragment.append(value.slice(start, i));
      const host = document.createElement("span"); host.innerHTML = equationHtml(latex, open === "[");
      fragment.append(...host.childNodes); start = end + 1; i = end;
    }
    if (found) { fragment.append(value.slice(start)); node.replaceWith(fragment); }
  }
}

export function typesetMath(root, output = "htmlAndMathml") {
  root.querySelectorAll("[data-classio-equation]").forEach(node => {
    const latex = node.getAttribute("data-latex") || "";
    node.classList.add("study-equation"); node.classList.toggle("is-display", node.getAttribute("data-display") === "block");
    node.setAttribute("role", "math"); node.setAttribute("aria-label", latex);
    node.innerHTML = katex.renderToString(latex, { displayMode: node.getAttribute("data-display") === "block", output, throwOnError: false, trust: false, maxExpand: 500, strict: "ignore" });
  });
}

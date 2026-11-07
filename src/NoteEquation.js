import { Node, mergeAttributes } from "@tiptap/core";
import katex from "katex";

export const NoteEquation = Node.create({
  name: "noteEquation", group: "inline", inline: true, atom: true,
  addAttributes() { return { latex: { default: "", parseHTML: element => element.getAttribute("data-latex") || "" } }; },
  parseHTML() { return [{ tag: "span[data-classio-equation]" }]; },
  renderHTML({ node, HTMLAttributes }) { return ["span", mergeAttributes(HTMLAttributes, { "data-classio-equation": "true", "data-latex": node.attrs.latex }), `\\(${node.attrs.latex}\\)`]; },
  addNodeView() { return ({ node }) => { const dom = document.createElement("span"); dom.className = "note-equation"; dom.setAttribute("role", "math"); dom.setAttribute("aria-label", node.attrs.latex); dom.innerHTML = katex.renderToString(node.attrs.latex, { throwOnError: false, trust: false, maxExpand: 500 }); return { dom }; }; },
});

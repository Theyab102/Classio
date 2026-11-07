import { markdownToNoteHtml, noteHtmlToMarkdown } from "./noteFormat";
import { typesetMath } from "./mathMarkdown";
jest.mock("./pdf", () => ({ openPdf: jest.fn() }));

test("physics notation survives Markdown parsing and becomes actual fractions, roots and subscripts", () => {
  const raw = String.raw`# Projectile motion

**Speed:** \(100\,\text{m/s}\), angle \(\theta\).

\[t=\sqrt{\frac{2\times500}{9.8}}\approx10.1\,\text{s}\]

- Horizontal: \(v_{0x}=v_0\cos\theta\)

| Quantity | Equation |
|---|---|
| Height | \(y=y_0-\frac{1}{2}gt^2\) |`;
  const host = document.createElement("div"); host.innerHTML = markdownToNoteHtml(raw);
  expect(host.querySelectorAll("[data-classio-equation]")).toHaveLength(5);
  expect(host.querySelector("h1").textContent).toBe("Projectile motion");
  expect(host.querySelector("strong").textContent).toBe("Speed:");
  const roundtrip = noteHtmlToMarkdown(host.innerHTML);
  expect(roundtrip).toContain(String.raw`\sqrt{\frac{2\times500}{9.8}}`);
  typesetMath(host);
  expect(host.querySelectorAll(".katex")).toHaveLength(5);
  expect(host.querySelector("mfrac")).not.toBeNull();
  expect(host.querySelector("msqrt")).not.toBeNull();
  expect(host.querySelector("msub")).not.toBeNull();
  expect(host.querySelector(".katex-error")).toBeNull();
});
test("dollar delimiters, old stripped delimiters and code are handled without guessing ordinary prose", () => {
  const raw = String.raw`Inline $F=ma$, display $$x=\frac{1}{2}gt^2$$.

Saved legacy equation (v_{x}=v_{0x}=\text{constant}).

Saved worked example (500=\tfrac{1}{2}(9.8)t^{2}), and speed (v_0).

Normal (a helpful explanation), [a reference].

` + "\n\n`\\(code\\)`";
  const host = document.createElement("div"); host.innerHTML = markdownToNoteHtml(raw);
  expect(host.querySelectorAll("[data-classio-equation]")).toHaveLength(5);
  typesetMath(host);
  expect(host.querySelector(".katex-error")).toBeNull();
  expect(host.textContent).toContain("(a helpful explanation)");
  expect(host.querySelector("code")).not.toBeNull();
});

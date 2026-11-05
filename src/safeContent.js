import DOMPurify from "dompurify";

const htmlEntities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, character => htmlEntities[character]);
}

export function sanitizeDocumentHtml(html) {
  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ["style", "form"],
    FORBID_ATTR: ["style", "srcset"],
  });
}

const key = "classio_access_v2";
const legacyKey = "classio_access_v1";
const valid = new Set(["eligible", "accepted", "blocked"]);

export function readAccess() {
  try {
    const record = JSON.parse(localStorage.getItem(key));
    if (valid.has(record?.decision)) {
      if (record.decision !== "blocked") return record.decision;
      const elapsed = Date.now() - record.at;
      return elapsed >= 0 && elapsed < 86400000 ? "blocked" : null;
    }
  } catch {}
  try { const decision = sessionStorage.getItem(legacyKey); return valid.has(decision) ? decision : null; }
  catch { return null; }
}

export function rememberAccess(decision) {
  if (!valid.has(decision)) return;
  // Store the decision only, never the date of birth or an AI request.
  try { localStorage.setItem(key, JSON.stringify({ decision, at: Date.now() })); } catch {}
  try { sessionStorage.setItem(legacyKey, decision); } catch {}
}

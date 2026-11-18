import { useEffect, useState } from "react";
import ChoiceMenu from "./ChoiceMenu";
import { aiMode, setAiMode } from "./aiPreferences";
import {
  useLocalAi,
  loadLocalModel,
  localCompatibility,
  disableLocalAi,
  clearLocalModels,
} from "./localAi";
import "./aiUpgrade.css";
const labels = {
  groq: "Groq",
  cloudflare: "Cloudflare",
  openrouter: "OpenRouter",
  ollama: "Private server AI",
};
export default function AISettings() {
  const [status, setStatus] = useState(null),
    [mode, setMode] = useState(aiMode),
    [consent, setConsent] = useState(false),
    [embedding, setEmbedding] = useState("english"),
    [error, setError] = useState("");
  const local = useLocalAi(),
    compatibility = localCompatibility();
  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/status", { signal: controller.signal })
      .then((response) => response.json())
      .then((data) => {
        setStatus(data);
        const allowed = [
          "auto",
          ...(data.providers || []).map((provider) =>
            typeof provider === "string" ? provider : provider.id,
          ),
          ...(data.browserLocalEnabled ? ["browser-local"] : []),
        ];
        if (!allowed.includes(aiMode())) {
          setAiMode("auto");
          setMode("auto");
        }
      })
      .catch(() => setStatus({ providers: [], aiConfigured: false }));
    return () => controller.abort();
  }, []);
  async function load(kind) {
    setError("");
    try {
      await loadLocalModel(kind, embedding);
    } catch (failure) {
      setError(failure.message);
    }
  }
  function disable() {
    disableLocalAi();
    if (aiMode() === "browser-local") {
      setAiMode("auto");
      setMode("auto");
    }
  }
  return (
    <div className="classio-ai-settings">
      <label>
        AI mode
        <ChoiceMenu
          aria-label="AI mode"
          value={mode}
          onChange={(event) => {
            setMode(event.target.value);
            setAiMode(event.target.value);
          }}
        >
          <option value="auto">Auto (Recommended)</option>
          {(status?.providers || []).map((provider) => {
            const id = typeof provider === "string" ? provider : provider.id;
            return (
              <option key={id} value={id}>
                {labels[id] || id}
                {provider.status === "temporarily-limited"
                  ? " · temporarily limited"
                  : ""}
              </option>
            );
          })}
          {status?.browserLocalEnabled && (
            <option value="browser-local">Local AI · on this device</option>
          )}
        </ChoiceMenu>
      </label>
      <p>
        {status === null
          ? "Checking available AI…"
          : status.aiConfigured
            ? "Auto chooses an available provider and recovers from temporary failures."
            : "Hosted AI is currently unavailable. Your writing and saved materials are kept."}
      </p>
      {status?.freeOnly && (
        <small>
          Free-only operation is enabled. Provider limits still apply.
        </small>
      )}
      {status?.browserLocalEnabled && (
        <details className="local-ai-options">
          <summary>On-device AI and semantic search</summary>
          <p>
            Model downloads are optional. Your study text stays on this device
            in Local AI mode.
          </p>
          <label className="studio-check">
            <input
              type="checkbox"
              checked={consent}
              onChange={(event) => setConsent(event.target.checked)}
            />
            Allow model downloads on this device
          </label>
          <p>
            Small local AI: about 400–600 MB to download; about 950 MB graphics
            memory.{" "}
            {compatibility.webgpu
              ? compatibility.memoryGB
                ? `Device memory: ${compatibility.memoryGB} GB.`
                : "Device memory is not reported."
              : "This browser does not support the required graphics feature."}
          </p>
          <div className="studio-controls">
            <button
              type="button"
              className="quiet-button"
              disabled={!consent || local.loading || !compatibility.canLoad}
              onClick={() => load("llm")}
            >
              {local.llm ? "Reload local model" : "Download local AI"}
            </button>
            <button type="button" className="quiet-button" onClick={disable}>
              Disable local AI
            </button>
          </div>
          <label>
            Semantic search model
            <ChoiceMenu
              aria-label="Semantic search model"
              value={embedding}
              onChange={(event) => setEmbedding(event.target.value)}
            >
              <option value="english">English · about 25 MB</option>
              <option value="multilingual">Multilingual · about 120 MB</option>
            </ChoiceMenu>
          </label>
          <p>
            Additional tokenizer/runtime files are downloaded. Regular search
            works without a model.
          </p>
          <button
            type="button"
            className="quiet-button"
            disabled={!consent || local.loading}
            onClick={() => load("embeddings")}
          >
            {local.embeddings
              ? "Reload semantic model"
              : "Enable semantic search"}
          </button>
          <button
            type="button"
            className="quiet-button"
            disabled={local.loading}
            onClick={() =>
              clearLocalModels()
                .then(() => {
                  if (aiMode() === "browser-local") {
                    setAiMode("auto");
                    setMode("auto");
                  }
                })
                .catch((failure) => setError(failure.message))
            }
          >
            Remove downloaded models
          </button>
          {local.loading && (
            <div role="status">
              Downloading/loading model: {local.progress}%
              <progress max="100" value={local.progress} />
            </div>
          )}
          {local.llm && (
            <p role="status">Local AI is ready for short text requests.</p>
          )}
          {local.embeddings && (
            <p role="status">Semantic search is ready in your library.</p>
          )}
        </details>
      )}
      {(error || local.error) && <p role="alert">{error || local.error}</p>}
    </div>
  );
}

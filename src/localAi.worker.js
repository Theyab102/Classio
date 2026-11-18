/* global globalThis */
let engine,
  extractor,
  extractorModel,
  scope = "",
  cancelled = new Set(),
  queue = Promise.resolve();
const MODEL = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
  embeddingModels = {
    english: "Xenova/all-MiniLM-L6-v2",
    multilingual: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
  };
const embeddings = new Map();
const check = (id) => {
  if (cancelled.has(id)) throw new DOMException("Cancelled", "AbortError");
};
globalThis.onmessage = ({ data }) => {
  if (data.type === "cancel") {
    cancelled.add(data.target);
    engine?.interruptGenerate();
    return;
  }
  queue = queue
    .catch(() => {})
    .then(async () => {
      const { id, type } = data;
      try {
        if (data.owner && scope !== data.owner) {
          scope = data.owner;
          embeddings.clear();
          await engine?.resetChat();
        }
        check(id);
        if (type === "load-llm") {
          if (!globalThis.navigator?.gpu) throw Error("WEBGPU_UNAVAILABLE");
          await engine?.unload();
          engine = null;
          const { CreateMLCEngine } = await import("@mlc-ai/web-llm");
          engine = await CreateMLCEngine(MODEL, {
            initProgressCallback: (report) =>
              globalThis.postMessage({
                id,
                progress: Math.round(report.progress * 100),
              }),
          });
          check(id);
          globalThis.postMessage({ id, result: true });
        } else if (type === "load-embeddings") {
          await extractor?.dispose?.();
          extractor = null;
          const { pipeline, env } = await import("@huggingface/transformers");
          env.allowLocalModels = false;
          env.backends.onnx.wasm.numThreads = 1;
          env.backends.onnx.wasm.wasmPaths =
            globalThis.location.origin + "/local-ai/";
          extractorModel =
            embeddingModels[data.model] || embeddingModels.english;
          extractor = await pipeline("feature-extraction", extractorModel, {
            device: "wasm",
            dtype: "q8",
            progress_callback: (report) => {
              if (report.progress != null)
                globalThis.postMessage({
                  id,
                  progress: Math.round(report.progress),
                });
            },
          });
          check(id);
          embeddings.clear();
          globalThis.postMessage({ id, result: true });
        } else if (type === "generate") {
          if (!engine) throw Error("LOCAL_NOT_READY");
          await engine.resetChat();
          let text = "";
          const stream = await engine.chat.completions.create({
            messages: data.payload.messages,
            max_tokens: Math.min(1600, data.payload.maxTokens || 1000),
            temperature: 0.35,
            stream: true,
          });
          for await (const part of stream) {
            check(id);
            const delta = part.choices?.[0]?.delta?.content || "";
            text += delta;
            globalThis.postMessage({ id, delta });
          }
          check(id);
          if (!text.trim()) throw Error("EMPTY_RESPONSE");
          globalThis.postMessage({ id, result: text });
        } else if (type === "search") {
          if (!extractor) throw Error("EMBEDDINGS_NOT_READY");
          const embed = async (text) => {
            const output = await extractor(text, {
              pooling: "mean",
              normalize: true,
            });
            return Array.from(output.data);
          };
          const query = await embed(data.query.slice(0, 1200)),
            hits = [];
          for (const item of data.documents.slice(0, 150)) {
            check(id);
            const fingerprint = item.id + ":" + item.text.slice(0, 5000),
              saved = embeddings.get(fingerprint);
            let vectors = saved;
            if (!vectors) {
              vectors = [];
              const text = item.text.slice(0, 4800);
              for (let start = 0; start < text.length; start += 1200) {
                check(id);
                vectors.push(await embed(text.slice(start, start + 1200)));
              }
              if (embeddings.size >= 150)
                embeddings.delete(embeddings.keys().next().value);
              embeddings.set(fingerprint, vectors);
            }
            hits.push({
              id: item.id,
              score: Math.max(
                -1,
                ...vectors.map((vector) =>
                  vector.reduce(
                    (sum, value, index) => sum + value * query[index],
                    0,
                  ),
                ),
              ),
            });
          }
          check(id);
          globalThis.postMessage({
            id,
            result: hits.sort((a, b) => b.score - a.score).slice(0, 30),
          });
        } else if (type === "clear-cache") {
          if (engine) await engine.unload();
          engine = null;
          await extractor?.dispose?.();
          extractor = null;
          embeddings.clear();
          // Pinned WebLLM uses these Cache API scopes. Its per-model delete
          // helper fetches a remote tensor manifest on a cache miss, so delete
          // the owned namespaces directly, including while offline.
          if (globalThis.caches)await Promise.all(['webllm/model','webllm/config','webllm/wasm','transformers-cache'].map(name=>globalThis.caches.delete(name)));
          globalThis.postMessage({ id, result: true });
        }
      } catch (error) {
        globalThis.postMessage({
          id,
          error:
            error.name === "AbortError"
              ? "CANCELLED"
              : type === "generate"
                ? "Local AI could not complete this request. Choose Auto or use a shorter section."
                : type === "search"
                  ? "Semantic search is unavailable. Regular search is still available."
                  : type==='clear-cache'?'Downloaded models could not all be removed. Try again or clear this site’s storage in your browser.':"The on-device model could not load. No study content was uploaded.",
        });
      } finally {
        cancelled.delete(id);
      }
    });
};

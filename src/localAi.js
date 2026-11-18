import { useSyncExternalStore } from "react";
import { aiOwner } from "./aiAccess";
let worker,
  workerPromise,
  epoch = 0,
  sequence = 0,
  state = {
    llm: false,
    embeddings: false,
    loading: false,
    progress: 0,
    error: "",
  };
const pending = new Map(),
  listeners = new Set();
function publish(patch) {
  state = { ...state, ...patch };
  listeners.forEach((notify) => notify());
}
export const localAiState = () => state;
export function useLocalAi() {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    localAiState,
    localAiState,
  );
}
async function getWorker() {
  if (worker) return worker;
  if (!workerPromise) {
    const started = epoch;
    workerPromise = import("./localAiWorkerFactory")
      .then(({ createLocalWorker }) => {
        if (started !== epoch)
          throw new DOMException("Local AI disabled", "AbortError");
        worker = createLocalWorker();
        worker.onmessage = ({ data }) => {
          const job = pending.get(data.id);
          if (!job) return;
          if (data.progress != null) {
            publish({ progress: data.progress });
            return;
          }
          if (data.delta) {
            job.onToken?.(data.delta);
            return;
          }
          pending.delete(data.id);
          job.cleanup();
          if (data.error)
            job.reject(
              data.error === "CANCELLED"
                ? new DOMException("Cancelled", "AbortError")
                : new Error(data.error),
            );
          else job.resolve(data.result);
        };
        worker.onerror = () => {
          disableLocalAi();
          publish({ error: "On-device AI stopped. Choose Auto to continue." });
        };
        return worker;
      })
      .catch((error) => {
        workerPromise = null;
        throw error;
      });
  }
  return workerPromise;
}
function request(type, data = {}, signal, onToken) {
  const owner = aiOwner(),
    id = ++sequence;
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Cancelled", "AbortError"));
      return;
    }
    const cancel = () => {
      worker?.postMessage({ type: "cancel", target: id });
      pending.delete(id);
      signal?.removeEventListener("abort", cancel);
      reject(new DOMException("Cancelled", "AbortError"));
    };
    signal?.addEventListener("abort", cancel, { once: true });
    pending.set(id, {
      resolve: (result) =>
        aiOwner() === owner
          ? resolve(result)
          : reject(new DOMException("Account changed", "AbortError")),
      reject,
      onToken: (delta) => {
        if (aiOwner() === owner) onToken?.(delta);
      },
      cleanup: () => signal?.removeEventListener("abort", cancel),
    });
    getWorker()
      .then((ready) => {
        if (pending.has(id)) ready.postMessage({ id, type, owner, ...data });
      })
      .catch((error) => {
        const job = pending.get(id);
        if (job) {
          pending.delete(id);
          job.cleanup();
          reject(error);
        }
      });
  });
}
export function localCompatibility() {
  return {
    webgpu: !!navigator.gpu,
    memoryGB: navigator.deviceMemory || null,
    canLoad:
      !!navigator.gpu &&
      (!navigator.deviceMemory || navigator.deviceMemory >= 4),
  };
}
export async function loadLocalModel(kind, model = "english") {
  if (kind === "llm" && !localCompatibility().canLoad)
    throw new Error(
      "This device does not have enough supported graphics/memory for local AI.",
    );
  publish({ loading: true, [kind]: false, progress: 0, error: "" });
  try {
    await request(kind === "llm" ? "load-llm" : "load-embeddings", { model });
    publish({ [kind]: true, progress: 100 });
  } catch (error) {
    publish({ error: error.message });
    throw error;
  } finally {
    publish({ loading: false });
  }
}
export async function generateLocal(payload, signal) {
  if (
    payload.audio ||
    payload.vision ||
    payload.messages?.some((message) => Array.isArray(message.content))
  )
    throw new Error(
      "The small local model supports text. Choose Auto for images or recordings.",
    );
  if (!state.llm)
    throw new Error(
      "Load the on-device model in Settings first, or choose Auto.",
    );
  if (
    JSON.stringify(payload.messages).length / 3 +
      Math.min(payload.maxTokens || 1000, 1600) >
    3500
  )
    throw new Error(
      "This section is too large for the small on-device model. Choose Auto or a shorter section.",
    );
  const { onToken, ...input } = payload;
  return request("generate", { payload: input }, signal, onToken);
}
export async function semanticSearch(query, documents, signal) {
  if (!state.embeddings) return null;
  return request("search", { query, documents }, signal);
}
export async function clearLocalModels() {
  disableLocalAi();
  publish({loading:true});
  try {
    await request("clear-cache");
  } finally {
    disableLocalAi();
  }
}
export function disableLocalAi() {
  epoch++;
  worker?.terminate();
  worker = null;
  workerPromise = null;
  for (const job of pending.values()) {
    job.cleanup();
    job.reject(new DOMException("Local AI disabled", "AbortError"));
  }
  pending.clear();
  publish({
    llm: false,
    embeddings: false,
    loading: false,
    progress: 0,
    error: "",
  });
}

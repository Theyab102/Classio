// Loaded only after an explicit on-device AI action.
export function createLocalWorker() {
  return new Worker(new URL("./localAi.worker.js", import.meta.url), {
    type: "module",
  });
}

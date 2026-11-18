import { useEffect, useState } from "react";
import { aiOwner } from "./aiAccess";
import { saveGuestBlob, getGuestBlob, removeGuestBlob } from "./guestWorkspace";
let database;
const active = new Set(),
  uuid = () =>
    window.crypto?.randomUUID?.() ||
    Date.now() + "-" + Math.random().toString(36).slice(2);
function open() {
  database ||= new Promise((resolve, reject) => {
    const request = indexedDB.open("classio_generation_jobs", 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore("jobs");
      request.result.createObjectStore("leases");
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(
        new Error(
          "Generation recovery storage is unavailable. Your existing work is kept.",
        ),
      );
  });
  return database;
}
async function transaction(name, mode, run) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(name, mode),
      request = run(tx.objectStore(name));
    tx.oncomplete = () => resolve(request?.result);
    tx.onerror = tx.onabort = () =>
      reject(
        new Error(
          "A generation checkpoint could not be saved. Your existing work is kept.",
        ),
      );
  });
}
const key = (owner, file, kind) => JSON.stringify([owner, file, kind]),
  guestId = (file, kind) => "generation-job:" + file + ":" + kind;
export const jobStorage = {
  async get(owner, file, kind) {
    if (owner.startsWith("guest:")) {
      const blob = await getGuestBlob(guestId(file, kind));
      return blob ? JSON.parse(await blob.text()) : null;
    }
    return transaction("jobs", "readonly", (store) =>
      store.get(key(owner, file, kind)),
    );
  },
  async put(job) {
    if (job.owner.startsWith("guest:"))
      return saveGuestBlob(
        guestId(job.file, job.kind),
        new Blob([JSON.stringify(job)], { type: "application/json" }),
      );
    return transaction("jobs", "readwrite", (store) =>
      store.put(job, key(job.owner, job.file, job.kind)),
    );
  },
  async claim(owner, file, kind, token) {
    const db = await open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction("leases", "readwrite"),
        store = tx.objectStore("leases"),
        request = store.get(key(owner, file, kind));
      let accepted = false;
      request.onsuccess = () => {
        const old = request.result;
        if (!old || old.until < Date.now() || old.token === token) {
          accepted = true;
          store.put(
            { token, until: Date.now() + 90000 },
            key(owner, file, kind),
          );
        }
      };
      tx.oncomplete = () =>
        accepted
          ? resolve()
          : reject(
              new Error(
                "This generation job is running in another tab. Wait briefly before resuming.",
              ),
            );
      tx.onerror = tx.onabort = () =>
        reject(new Error("The generation job could not be locked."));
    });
  },
  async release(owner, file, kind, token) {
    const db = await open();
    return new Promise((resolve) => {
      const tx = db.transaction("leases", "readwrite"),
        store = tx.objectStore("leases"),
        request = store.get(key(owner, file, kind));
      request.onsuccess = () => {
        if (request.result?.token === token)
          store.delete(key(owner, file, kind));
      };
      tx.oncomplete = tx.onerror = tx.onabort = () => resolve();
    });
  },
};
export async function readGenerationJob(file, kind) {
  const owner = aiOwner(),
    local = await jobStorage.get(owner, file.id, kind),
    hint = file.generationJobs?.[kind];
  const saved =
    [local, hint]
      .filter((item) => item?.owner === owner)
      .sort((a, b) => b.updatedAt - a.updatedAt)[0] || null;
  return aiOwner() === owner ? saved : null;
}
export function useGenerationJob(file, kind) {
  const hint = file.generationJobs?.[kind],
    safeHint = hint?.owner === aiOwner() ? hint : null;
  const [job, setJob] = useState(safeHint);
  const fileId = file.id;
  useEffect(() => {
    let live = true;
    readGenerationJob(
      { id: fileId, generationJobs: { [kind]: safeHint } },
      kind,
    )
      .then((saved) => {
        if (live) setJob(saved);
      })
      .catch(() => {
        if (live) setJob(safeHint);
      });
    return () => {
      live = false;
    };
  }, [fileId, kind, safeHint]);
  return job?.owner === aiOwner() ? job : null;
}
export function checkpointPatch(file, kind, job) {
  return { ...file, generationJobs: { ...file.generationJobs, [kind]: job } };
}
export const resumable = (job) => job && job.status !== "completed";
export function distinct(
  items,
  identity = (item) => (typeof item === "string" ? item : JSON.stringify(item)),
) {
  const seen = new Set();
  for (const item of items) {
    const value = identity(item)
      .normalize("NFKC")
      .toLowerCase()
      .replace(/\s+/g, " ")
      .trim();
    if (value && seen.has(value))
      throw new Error(
        "The AI repeated completed content. Saved sections are kept; resume to try this section again.",
      );
    seen.add(value);
  }
  return items;
}
export async function runCheckpointJob({
  fileId,
  kind,
  input,
  resume,
  plan,
  step,
  total,
  assemble = (job) => job.sections,
  onCheckpoint = () => {},
  onProgress = () => {},
  signal,
  store = jobStorage,
}) {
  const owner = aiOwner(),
    lockKey = key(owner, fileId, kind),
    token = uuid();
  if (active.has(lockKey))
    throw new Error("This generation job is already running.");
  active.add(lockKey);
  let job,
    claimed = false;
  const check = () => {
    if (signal?.aborted || aiOwner() !== owner)
      throw new DOMException("Generation paused", "AbortError");
  };
  const save = async () => {
    check();
    job.updatedAt = Date.now();
    if (store.claim) await store.claim(owner, fileId, kind, token);
    await store.put(JSON.parse(JSON.stringify(job)));
    check();
    onCheckpoint(JSON.parse(JSON.stringify(job)));
  };
  try {
    if (store.claim) {
      await store.claim(owner, fileId, kind, token);
      claimed = true;
    }
    const saved = resume ? await store.get(owner, fileId, kind) : null;
    if (resume) {
      const candidate =
        saved && saved.updatedAt >= resume.updatedAt ? saved : resume;
      if (candidate.id !== resume.id)
        throw new Error(
          "A newer generation job is saved for this file. Reopen the material to resume its current draft.",
        );
      if (
        candidate.owner !== owner ||
        candidate.file !== fileId ||
        candidate.kind !== kind ||
        candidate.version !== 1
      )
        throw new Error(
          "This generation checkpoint does not belong to the current file and account.",
        );
      if (candidate.status === "completed")
        throw new Error("This generated draft is already saved in your file.");
      job = JSON.parse(JSON.stringify(candidate));
    } else
      job = {
        version: 1,
        id: uuid(),
        owner,
        file: fileId,
        kind,
        input,
        sections: [],
        outline: null,
        status: "running",
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
    job.status = "running";
    job.error = "";
    delete job.failure;
    await save();
    if (plan && !job.outline) {
      onProgress("Planning and saving the outline…");
      job.outline = await plan(job, signal);
      check();
      await save();
    }
    job.total = typeof total === "function" ? total(job) : total;
    if (!Number.isInteger(job.total) || job.total < 1 || job.total > 100)
      throw new Error("The generation plan is outside the supported size.");
    await save();
    for (let index = job.sections.length; index < job.total; index++) {
      check();
      onProgress(
        `Writing section ${index + 1} of ${job.total} · ${index} saved`,
      );
      let result;
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          result = await step(index, job, signal, attempt);
          check();
          break;
        } catch (error) {
          if (
            signal?.aborted ||
            error.name === "AbortError" ||
            aiOwner() !== owner ||
            attempt === 1 ||
            error.status === 429 ||
            error.status === 401 ||
            error.status === 403 ||
            [
              "AUTH_UNAVAILABLE",
              "AUTH_REQUIRED",
              "AUTH_EXPIRED",
              "AUTH_REVOKED",
              "GUEST_SESSION_REQUIRED",
            ].includes(error.code) ||
            error.code === "CONTENT_BLOCKED" ||
            error.code === "AI_NOT_CONFIGURED"
          )
            throw error;
          onProgress("Retrying this unfinished section…");
          await new Promise((resolve) => setTimeout(resolve, 200));
          check();
        }
      }
      job.sections[index] = result;
      await save();
    }
    const result = await assemble(job);
    check();
    job.status = "ready";
    await save();
    return result;
  } catch (error) {
    if (job && aiOwner() === owner) {
      job.status =
        signal?.aborted || error.name === "AbortError" ? "paused" : "failed";
      job.error =
        !job.sections.length && !job.outline
          ? "Generation has not started. Your existing work is kept."
          : job.status === "paused"
            ? "Generation paused. Completed sections are saved."
            : "This section could not finish. Completed sections are saved.";
      job.updatedAt = Date.now();
      job.failure = {
        code:
          typeof error.code === "string"
            ? error.code.slice(0, 60)
            : "GENERATION_FAILED",
        status: Number(error.status) || 0,
      };
      try {
        await store.put(JSON.parse(JSON.stringify(job)));
        onCheckpoint(JSON.parse(JSON.stringify(job)));
      } catch {
        /* Do not replace a storage/provider error with a false recovery claim. */
      }
    }
    throw error;
  } finally {
    active.delete(lockKey);
    if (claimed) await store.release?.(owner, fileId, kind, token);
  }
}
// A ready draft stays recoverable until its caller has actually inserted it.
export function appliedGenerationPatch(file, kind, job) {
  if (!job || job.owner !== aiOwner()) return {};
  const completed = { ...job, status: "completed", updatedAt: Date.now() + 1 };
  jobStorage.put(completed).catch(() => {});
  return { generationJobs: { ...file.generationJobs, [kind]: completed } };
}
export async function deleteGenerationJobs(file) {
  const owner = aiOwner();
  for (const kind of [
    "presentation",
    "notes",
    "course",
    "quiz",
    "flashcards",
  ]) {
    if (owner.startsWith("guest:")) await removeGuestBlob(guestId(file, kind));
    else
      await transaction("jobs", "readwrite", (store) =>
        store.delete(key(owner, file, kind)),
      );
  }
}

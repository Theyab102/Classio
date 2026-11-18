import { resumable } from "./generationJobs";
import "./aiUpgrade.css";
export default function GenerationRecovery({
  job,
  busy,
  onResume,
  onUseDraft,
}) {
  if (!resumable(job)) return null;
  const saved = job.sections?.length || 0;
  return (
    <aside
      className="generation-recovery"
      aria-label="Saved generation progress"
    >
      <strong>Generation progress saved</strong>
      <p>
        {saved} of {job.total || "planned"} sections saved.{" "}
        {busy
          ? "New sections save automatically."
          : job.status === "paused"
            ? "Generation is paused."
            : "Continue from the next unfinished section."}
      </p>
      <div className="studio-controls">
        {!busy && (
          <button
            type="button"
            className="quiet-button is-primary"
            onClick={() => onResume(job)}
          >
            Resume generation
          </button>
        )}
        {!busy && saved > 0 && onUseDraft && (
          <button
            type="button"
            className="quiet-button"
            onClick={() => onUseDraft(job)}
          >
            Use saved draft
          </button>
        )}
      </div>
    </aside>
  );
}

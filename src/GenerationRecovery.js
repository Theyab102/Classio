import { resumable } from "./generationJobs";
import "./aiUpgrade.css";
import useRetryDelay from './useRetryDelay';
export default function GenerationRecovery({
  job,
  busy,
  onResume,
  onUseDraft,
}) {
  const wait = useRetryDelay(job?.failure?.retryAt);
  if (!resumable(job)) return null;
  const saved = job.sections?.length || 0;
  if (busy && !saved) return null;
  if (!saved && !job.outline)
    return (
      <div className="generation-retry" role="status">
        <span>{job.status === 'failed' ? 'No new sections were completed. Your existing work is kept.' : 'Generation hasn’t started yet.'}</span>
        <button
          type="button"
          className="text-button"
          disabled={busy || wait > 0}
          onClick={() => onResume(job)}
        >
          {wait ? `Try again in ${wait}s` : 'Try again'}
        </button>
      </div>
    );
  return (
    <aside
      className="generation-recovery"
      aria-label="Saved generation progress"
    >
      <strong>Generation progress saved</strong>
      <p>
        {saved
          ? `${saved} of ${job.total || "planned"} sections saved.`
          : "Outline saved. No content sections have been completed yet."}{" "}
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
            disabled={wait > 0}
            onClick={() => onResume(job)}
          >
            {wait ? `Resume in ${wait}s` : 'Resume generation'}
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

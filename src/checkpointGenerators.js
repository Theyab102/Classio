import { runCheckpointJob, distinct } from "./generationJobs";
import { parseAiJson } from "./aiJson";
import { parseDeckResponse } from "./DeckDesign";
import { courseSections } from "./courseGeneration";
import { sendStudyRequest } from "./studyTools";
import { languageInstruction } from "./studyContext";
import { MATH_OUTPUT_GUIDE } from "./mathMarkdown";

function deck(raw, count) {
  const value = parseAiJson(raw);
  if (!Array.isArray(value.slides) || value.slides.length !== count)
    throw new Error(
      "The presentation has missing or extra slides. Saved slides are kept.",
    );
  const parsed = parseDeckResponse(raw, count);
  distinct(parsed.slides, (slide) => slide.title);
  return parsed;
}
export function generatePresentation({
  input,
  resume,
  send = sendStudyRequest,
  ...options
}) {
  const frozen = resume?.input || input,
    large = frozen.count > 8;
  return runCheckpointJob({
    ...options,
    kind: "presentation",
    input: frozen,
    resume,
    total: large ? frozen.count : 1,
    plan: large
      ? async (job, signal) => {
          const raw = await send(
            {
              ...frozen.request,
              messages: [
                {
                  role: "system",
                  content: `Plan exactly ${frozen.count} distinct educational slides. Return JSON {title, designs, outline:[{title,focus,type}]}. Include design options following these instructions, but no slide bodies yet. ${frozen.request.messages[0].content}\nFINAL OUTPUT OVERRIDE: Return only {title,designs,outline:[{title,focus,type}]}, with exactly ${frozen.count} outline entries. Do not return slides or slide bodies yet.`,
                },
                { role: "user", content: frozen.request.messages[1].content + '\nNumbered source sections (keep outline coverage in this order):\n' + courseSections(frozen.source||'',frozen.count).map((part,index)=>`${index+1}: ${part.slice(0,650)}`).join('\n') },
              ],
              maxTokens: 3600,
            },
            signal,
          );
          const plan = parseAiJson(raw);
          if (
            !Array.isArray(plan.outline) ||
            plan.outline.length !== frozen.count ||
            plan.outline.some(
              (item) =>
                !item ||
                typeof item.title !== "string" ||
                !item.title.trim() ||
                item.title.length > 200 ||
                typeof item.focus !== "string" ||
                !item.focus.trim() ||
                item.focus.length > 600,
            )
          )
            throw new Error("The presentation outline is incomplete.");
          distinct(plan.outline, (item) => item.title);
          return {
            title: String(plan.title || frozen.topic).slice(0, 200),
            designs: deck(
              JSON.stringify({
                title: plan.title,
                designs: plan.designs,
                slides: [{ type: "title", title: plan.title || frozen.topic }],
              }),
              1,
            ).designs,
            outline: plan.outline,
          };
        }
      : undefined,
    step: async (index, job, signal, attempt) => {
      if (!large) {
        const result = deck(await send(frozen.request, signal), frozen.count);
        result.slides = result.slides.map((slide, i) => ({
          ...slide,
          id: job.id + ":" + i,
        }));
        return result;
      }
      const assigned = courseSections(frozen.source || "", frozen.count)[index];
      const raw = await send(
        {
          ...frozen.request,
          messages: [
            {
              role: "system",
              content: `Write ONLY slide ${index + 1} from the saved outline, as JSON {title,slides:[one slide]}. Ignore the full-deck count above; return exactly ONE slide. Keep at most ${attempt ? 3 : 4} concise bullets and move detail to note. Do not repeat earlier slides. ${frozen.request.messages[0].content}\nFINAL OVERRIDE: Exactly one slide, no designs. ${MATH_OUTPUT_GUIDE}`,
            },
            {
              role: "user",
              content: `Topic: ${frozen.topic}\nOutline: ${JSON.stringify(job.outline.outline)}\nAssigned slide: ${JSON.stringify(job.outline.outline[index])}\nSource section:\n${assigned.slice(0, 14000)}\nFigure catalog:\n${frozen.figureGuide || "None"}\nAlready saved slide titles: ${job.sections.map((slide) => slide.title).join("; ")}`,
            },
          ],
          maxTokens: attempt ? 1000 : 1800,
        },
        signal,
      );
      const slide = deck(raw, 1).slides[0];
      distinct([...job.sections, slide], (item) => item.title);
      return { ...slide, id: job.id + ":" + index };
    },
    assemble: (job) =>
      large
        ? {
            title: job.outline.title,
            designs: job.outline.designs,
            slides: job.sections,
          }
        : job.sections[0],
  });
}

function validateNotes(text, previous) {
  if (typeof text !== "string" || !text.trim() || text.length > 50000)
    throw new Error("The notes section is empty or too large.");
  const paragraphs = text
    .split(/\n\s*\n/)
    .filter(
      (part) => part.trim().length > 80 && !/^\s*(?:[#|]|\$\$)/.test(part),
    );
  distinct([
    ...previous.flatMap((part) =>
      part
        .split(/\n\s*\n/)
        .filter((p) => p.trim().length > 80 && !/^\s*(?:[#|]|\$\$)/.test(p)),
    ),
    ...paragraphs,
  ]);
  return text.trim();
}
export function generateNotes({
  input,
  resume,
  send = sendStudyRequest,
  ...options
}) {
  const frozen = resume?.input || input,
    count = Math.max(
      1,
      Math.min(12, Math.ceil((frozen.source?.length || 0) / 10000)),
    );
  const sections = courseSections(frozen.source || "", count);
  return runCheckpointJob({
    ...options,
    kind: "notes",
    input: frozen,
    resume,
    total: count,
    plan:
      count > 1
        ? async (job, signal) => {
            const outline = parseAiJson(
              await send(
                {
                  task: "notes",
                  messages: [
                    {
                      role: "system",
                      content: `Plan exactly ${count} distinct study-note sections in source order. Return JSON array of {title,focus}. Do not write the notes yet. ${languageInstruction(frozen.source)}`,
                    },
                    {
                      role: "user",
                      content:
                        `Topic: ${frozen.topic}\n` +
                        sections
                          .map(
                            (part, i) =>
                              `Section ${i + 1}: ${part.slice(0, 1100)}`,
                          )
                          .join("\n"),
                    },
                  ],
                  maxTokens: 1600,
                },
                signal,
              ),
            );
            if (
              !Array.isArray(outline) ||
              outline.length !== count ||
              outline.some(
                (item) =>
                  !item ||
                  typeof item.title !== "string" ||
                  !item.title.trim() ||
                  typeof item.focus !== "string" ||
                  !item.focus.trim() ||
                  item.focus.length > 500,
              )
            )
              throw new Error("The notes outline is incomplete.");
            distinct(outline, (item) => item.title);
            return outline;
          }
        : undefined,
    step: async (index, job, signal, attempt) => {
      const request =
        count === 1
          ? frozen.request
          : {
              ...frozen.request,
              messages: [
                frozen.request.messages[0],
                {
                  role: "user",
                  content: `Topic: ${frozen.topic}\nSaved outline: ${JSON.stringify(job.outline)}\nWrite ONLY section ${index + 1}: ${JSON.stringify(job.outline[index])}. Use H2 for this section, a H1 title only on section 1. Do not repeat any other section. Keep this section under ${attempt ? 1800 : 3200} characters.\nAssigned source section:\n${sections[index]}\nAlready covered: ${job.outline
                    .slice(0, index)
                    .map((item) => item.title)
                    .join("; ")}`,
                },
              ],
              maxTokens: attempt ? 1600 : 3000,
            };
      return validateNotes(await send(request, signal), job.sections);
    },
    assemble: (job) => job.sections.join("\n\n"),
  });
}

export function generateStudySet({
  kind,
  input,
  resume,
  parse,
  send = sendStudyRequest,
  ...options
}) {
  const frozen = resume?.input || input,
    size = kind === "quiz" ? 8 : 10;
  if (!Number.isInteger(frozen.count) || frozen.count < 1 || frozen.count > 50)
    throw new Error("Choose a whole number from 1 to 50.");
  return runCheckpointJob({
    ...options,
    kind,
    input: frozen,
    resume,
    total: Math.ceil(frozen.count / size),
    step: async (index, job, signal, attempt) => {
      const batch = Math.min(size, frozen.count - index * size),
        previous = job.sections.flat(),
        short = frozen.type === "short-answer";
      const format =
        kind === "quiz"
          ? short
            ? 'question, type ("short-answer"), answer (a concise expected answer), acceptedAnswers (one to five concise equivalent answers), hint, explanation'
            : "question, options (four strings), correct (zero-based index), hint, explanation"
          : "question and answer";
      const raw = await send(
        {
          task: kind === "quiz" ? "questions" : "flashcards",
          messages: [
            {
              role: "system",
              content: `Create exactly ${batch} distinct ${frozen.difficulty || "standard"} ${kind === "quiz" ? (short ? "short-answer study questions" : "multiple-choice study questions") : "study flashcards"}. Return only a JSON array with ${format}. ${languageInstruction(frozen.source)} Use source evidence when supplied; otherwise teach the topic without claiming document evidence. ${MATH_OUTPUT_GUIDE} ${attempt ? "Keep every explanation concise." : ""}`,
            },
            {
              role: "user",
              content: `Topic: ${frozen.topic}\nSource:\n${frozen.source}\nAvoid repeating: ${[...(frozen.avoid || []), ...previous.map((item) => item.question)].join("; ").slice(0, 10000)}`,
            },
          ],
          maxTokens: Math.min(
            7000,
            batch * (kind === "quiz" ? 550 : 300) + 500,
          ),
        },
        signal,
      );
      const rows = parse(raw, batch);
      if (rows.length !== batch)
        throw new Error(
          "The AI returned a different count. Your saved questions are kept.",
        );
      distinct([...previous, ...rows], (item) => item.question);
      if (frozen.avoid?.length)
        distinct(
          [...frozen.avoid.map((question) => ({ question })), ...rows],
          (item) => item.question,
        );
      return rows.map((row, i) => ({
        ...row,
        id: job.id + ":" + (index * size + i),
      }));
    },
    assemble: (job) => job.sections.flat(),
  });
}

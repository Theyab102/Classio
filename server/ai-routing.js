// Workload choices are server-owned; clients cannot select arbitrary models.
function workload(input) {
  if (input.audio) return "transcription";
  if (input.vision) return "vision";
  if (["classification", "style", "avatar-fit"].includes(input.task))
    return "fast";
  if (
    ["notes", "questions", "flashcards", "presentation"].includes(input.task) ||
    input.maxTokens > 1600
  )
    return "study";
  return "conversation";
}
function modality(input) {
  return input.audio ? "audio" : input.vision ? "vision" : "text";
}
function configuredModel(provider, input, env) {
  const lane = workload(input),
    prefix = "AI_" + provider.toUpperCase();
  if (lane === "transcription") return env[prefix + "_AUDIO_MODEL"];
  if (lane === "vision")
    return (
      env[prefix + "_VISION_MODEL"] ||
      (provider === "groq" ? env.GROQ_VISION_MODEL : undefined)
    );
  return (
    env[prefix + (lane === "study" ? "_STUDY_MODEL" : "_FAST_MODEL")] ||
    env[prefix + "_MODEL"] ||
    (provider === "groq" ? env.GROQ_TEXT_MODEL : undefined)
  );
}
module.exports = { workload, modality, configuredModel };

// Workload choices are server-owned; clients cannot select arbitrary models.
function workload(input) {
  if (input.audio) return "transcription";
  if (input.vision) return "vision";
  if (["classification", "style", "avatar-fit", "planning"].includes(input.task))
    return "fast";
  const latest=[...(input.messages||[])].reverse().find(message=>message.role==='user')?.content;
  const question=typeof latest==='string'?latest:Array.isArray(latest)?latest.filter(part=>part.type==='text').map(part=>part.text).join(' '):'';
  const mathematics=/\b(?:solve|simplif\w*|factori[sz]\w*|polynomial|quadratic|algebra|equation|end behavior|step.by.step)\b|\b[fx]\s*\(/i.test(question);
  if (
    mathematics || ["notes", "questions", "flashcards", "presentation", "podcast"].includes(input.task) ||
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

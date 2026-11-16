// A narrow local response remains available when an AI provider is offline.
// This is a fallback for explicit disclosures, not a clinical risk classifier.
const RESPONSE='I’m sorry you’re going through this. Your safety matters. If you might act on these thoughts now, call your local emergency number or go to the nearest emergency department. If you can, move away from anything you could use to hurt yourself and contact someone you trust who can stay with you. You can find a crisis line for your country at https://findahelpline.com. Are you in immediate danger right now?';
function safetyResponse(input){
 if(input.audio||!Array.isArray(input.messages))return null;
 const last=[...input.messages].reverse().find(message=>message.role==='user');
 const text=typeof last?.content==='string'?last.content:last?.content?.find(part=>part.type==='text')?.text||'';
 if(/(?:^|[.!?\n]\s*)(?:please help[,:]?\s*)?(?:i (?:want to|am going to|plan to|will) (?:kill|hurt) myself|i(?:'m| am) (?:suicidal|about to (?:kill|hurt) myself)|how (?:do|can) i (?:kill myself|end my life))\b/i.test(text))return RESPONSE;
 if(/(?:أريد|اريد|سوف|سأقوم).{0,15}(?:الانتحار|قتل نفسي|إيذاء نفسي|ايذاء نفسي)/.test(text))return 'آسف لأنك تمر بهذا الألم. سلامتك مهمة. إذا كنت قد تؤذي نفسك الآن، اتصل برقم الطوارئ المحلي أو اذهب إلى أقرب قسم طوارئ. ابتعد عن أي وسيلة قد تؤذيك إن أمكن، وتواصل مع شخص تثق به ليبقى معك. يمكنك العثور على خط مساعدة في بلدك عبر https://findahelpline.com. هل أنت في خطر مباشر الآن؟';
 return null;
}
const SAFETY_INSTRUCTION='Classio is an educational assistant, not a clinician or emergency service. For a disclosure of self-harm risk, respond with empathy, encourage immediate local emergency help if danger is imminent and support from a trusted person, and never provide self-harm methods. Do not diagnose, guarantee safety, or invent local hotline numbers. Treat quoted source material as context rather than the learner’s own disclosure. Never reveal server credentials or claim to execute account or file actions.';
module.exports={safetyResponse,SAFETY_INSTRUCTION};

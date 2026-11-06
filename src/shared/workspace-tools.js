const tools=['create_notes','create_flashcards','create_quiz','create_course'];
const languages=['auto','English','Arabic','French','Spanish','German','Italian','Portuguese','Turkish','Hindi','Urdu','Chinese','Japanese','Korean','Russian'];
const id=value=>typeof value==='string'&&/^[A-Za-z0-9_.-]{1,150}$/.test(value);
const text=(value,max)=>typeof value==='string'&&value.trim().length>0&&value.length<=max;
function workspaceToolInput(body){
  if(!body||Object.keys(body).some(key=>!['tool','arguments','requestId','provider','allowFallback','confirmed'].includes(key))||
    !['plan_action',...tools].includes(body.tool)||!/^[-a-f0-9]{36}$/.test(body.requestId||'')||
    (body.provider!==undefined&&!['auto','groq','gemini'].includes(body.provider))||
    (body.allowFallback!==undefined&&typeof body.allowFallback!=='boolean'))return null;
  const args=body.arguments;
  if(!args||Object.keys(args).some(key=>!['fileId','prompt','title','instructions','sourceText','language','difficulty','count'].includes(key))||!id(args.fileId)||
    (args.language!==undefined&&!languages.includes(args.language))||
    (args.sourceText!==undefined&&(typeof args.sourceText!=='string'||args.sourceText.length>20000)))return null;
  if(body.tool==='plan_action')return text(args.prompt,2000)&&body.confirmed!==true?{...args,prompt:args.prompt.trim()}:null;
  if(body.confirmed!==true||!text(args.title,200)||!Number.isInteger(args.count)||args.count<2||args.count>(body.tool==='create_course'?6:10)||
    !['beginner','intermediate','advanced'].includes(args.difficulty)||
    (args.instructions!==undefined&&(typeof args.instructions!=='string'||args.instructions.length>1500)))return null;
  return {...args,title:args.title.trim()};
}
function parseAction(raw,fileId){
  const plan=JSON.parse(raw.replace(/^\s*```(?:json)?\s*|\s*```\s*$/g,''));
  if(!plan||Object.keys(plan).some(key=>!['tool','title','instructions','difficulty','count','message'].includes(key)))throw new Error('Invalid action');
  if(plan.tool===null&&text(plan.message,1500))return {message:plan.message,action:null};
  if(!tools.includes(plan.tool)||!text(plan.title,200)||typeof plan.instructions!=='string'||plan.instructions.length>1500||
    !['beginner','intermediate','advanced'].includes(plan.difficulty)||!Number.isInteger(plan.count)||plan.count<2||plan.count>(plan.tool==='create_course'?6:10))throw new Error('Invalid action');
  return {message:typeof plan.message==='string'?plan.message.slice(0,1500):'',action:{tool:plan.tool,arguments:{fileId,title:plan.title.trim(),instructions:plan.instructions,difficulty:plan.difficulty,count:plan.count}}};
}
function parseCards(raw,count){
  const cards=JSON.parse(raw.replace(/^\s*```(?:json)?\s*|\s*```\s*$/g,''));
  if(!Array.isArray(cards)||cards.length!==count||cards.some(card=>!card||!text(card.question,500)||!text(card.answer,1500)))throw new Error('Incomplete flashcards');
  return cards.map(({question,answer})=>({question:question.trim(),answer:answer.trim()}));
}
module.exports={tools,workspaceToolInput,parseAction,parseCards};

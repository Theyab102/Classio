const voices = ['aria','nova','jade','echo','atlas','fable'];
function podcastInput(value) {
  if(!value || Object.keys(value).some(key=>!['dialogue','voices','language','speed'].includes(key)) || value.language!=='en' ||
    !Array.isArray(value.voices)||![1,2].includes(value.voices.length)||new Set(value.voices).size!==value.voices.length||value.voices.some(voice=>!voices.includes(voice))||
    typeof value.speed!=='number'||!Number.isFinite(value.speed)||value.speed<0.75||value.speed>1.25||Math.abs(value.speed*100-Math.round(value.speed*100))>1e-8||
    !Array.isArray(value.dialogue)||value.dialogue.length<1||value.dialogue.length>80) return null;
  const dialogue=[];
  for(const turn of value.dialogue){
    if(!turn||Object.keys(turn).some(key=>!['speaker','text'].includes(key))||!Number.isInteger(turn.speaker)||turn.speaker<0||turn.speaker>=value.voices.length||typeof turn.text!=='string')return null;
    const text=turn.text.replace(/\s+/g,' ').trim();
    if(text.length<3||text.length>1200)return null;
    dialogue.push({speaker:turn.speaker,text});
  }
  if(dialogue.reduce((sum,turn)=>sum+turn.text.length,0)>40000)return null;
  if(value.voices.length===2 && ![0,1].every(speaker=>dialogue.some(turn=>turn.speaker===speaker)))return null;
  return {language:'en',voices:[...value.voices],speed:value.speed,dialogue};
}
function podcastFingerprint(input){return [input.language,input.voices.join(','),String(Math.round(input.speed*100)),...input.dialogue.map(turn=>turn.speaker+':'+turn.text)].join('\n');}
module.exports={podcastInput,podcastFingerprint,voices};

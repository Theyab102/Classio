// Adaptive energy-based activity detection runs locally, without sending audio.
export function createSilenceDetector({silenceMs=2000,threshold=.018,maxMs=45000,quietMs=12000}={}){
  let first,activeSince,lastVoice,heard=false,floor=.004;
  return{sample(rms,now){first??=now;const energy=Number.isFinite(rms)?Math.max(0,rms):0;if(energy<threshold)floor=floor*.97+energy*.03;
    const speaking=energy>Math.max(threshold,Math.min(.055,floor*2.8));if(speaking){activeSince??=now;if(now-activeSince>=150){heard=true;lastVoice=now;}}else activeSince=undefined;
    if(now-first>=maxMs)return{stop:true,reason:heard?'limit':'no-speech',heard};if(heard&&!speaking&&now-lastVoice>=silenceMs)return{stop:true,reason:'silence',heard};if(!heard&&now-first>=quietMs)return{stop:true,reason:'no-speech',heard};return{stop:false,speaking,heard,level:Math.min(1,energy/.12)};
  }};
}
export function playListeningTone(context){if(context.state!=='running')return;const oscillator=context.createOscillator(),gain=context.createGain();oscillator.type='sine';oscillator.frequency.setValueAtTime(660,context.currentTime);oscillator.frequency.linearRampToValueAtTime(880,context.currentTime+.1);gain.gain.setValueAtTime(0,context.currentTime);gain.gain.linearRampToValueAtTime(.035,context.currentTime+.025);gain.gain.exponentialRampToValueAtTime(.001,context.currentTime+.14);oscillator.connect(gain);gain.connect(context.destination);oscillator.start();oscillator.stop(context.currentTime+.15);oscillator.onended=()=>{oscillator.disconnect();gain.disconnect();};}

// AbortSignal.any arrived after Safari's AbortController support. Keep podcast
// cancellation working on devices that have the controller but not this method.
export function anySignal(signals){
 const active=signals.filter(Boolean);
 if(AbortSignal.any)return AbortSignal.any(active);
 const controller=new AbortController(),listeners=[];
 const cancel=signal=>{controller.abort(signal.reason);for(const [source,listener] of listeners)source.removeEventListener('abort',listener);};
 for(const signal of active){if(signal.aborted){cancel(signal);break;}const listener=()=>cancel(signal);listeners.push([signal,listener]);signal.addEventListener('abort',listener,{once:true});}
 return controller.signal;
}

import {anySignal} from './abortSignal';
test('signal composition works on browsers without AbortSignal.any',()=>{
 const descriptor=Object.getOwnPropertyDescriptor(AbortSignal,'any');Object.defineProperty(AbortSignal,'any',{value:undefined,configurable:true});
 try{const first=new AbortController(),second=new AbortController(),signal=anySignal([undefined,first.signal,second.signal]);expect(signal.aborted).toBe(false);second.abort();expect(signal.aborted).toBe(true);expect(first.signal.aborted).toBe(false);expect(anySignal([second.signal]).aborted).toBe(true);}
 finally{if(descriptor)Object.defineProperty(AbortSignal,'any',descriptor);else delete AbortSignal.any;}
});

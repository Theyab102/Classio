import { useCallback, useEffect, useRef, useState } from "react";

export function historyReducer(history, action) {
  if (action.type === "undo") return history.past.length ? { past: history.past.slice(0,-1), present: history.past.at(-1), future: [history.present,...history.future] } : history;
  if (action.type === "redo") return history.future.length ? { past: [...history.past,history.present], present: history.future[0], future: history.future.slice(1) } : history;
  const value = typeof action.value === "function" ? action.value(history.present) : action.value;
  if (JSON.stringify(value) === JSON.stringify(history.present)) return history;
  return { past: [...history.past,history.present].slice(-40), present: value, future: [] };
}
export default function useActionHistory(initial, scope, resetKey) {
  const [history,setHistory] = useState(() => ({past:[],present:typeof initial === "function" ? initial() : initial,future:[]}));
  const previousKey = useRef(resetKey);
  useEffect(() => { if(previousKey.current !== resetKey){ previousKey.current=resetKey; setHistory({past:[],present:typeof initial === "function"?initial():initial,future:[]}); } },[resetKey,initial]);
  const current = useRef(history); current.current = history;
  const dispatch = useCallback(action => setHistory(h => historyReducer(h,action)),[]);
  useEffect(() => {
    if (!scope?.current) return;
    const node = scope.current;
    const keys = event => {
      if (!(event.ctrlKey || event.metaKey) || event.altKey || !node.contains(event.target) || event.target.closest('input,textarea,[contenteditable=true]')) return;
      const key = event.key.toLowerCase(), redo = key === "y" || (key === "z" && event.shiftKey);
      if (key !== "z" && key !== "y") return;
      if (redo ? !current.current.future.length : !current.current.past.length) return;
      event.preventDefault(); event.stopPropagation(); dispatch({type:redo?"redo":"undo"});
    };
    node.addEventListener("keydown",keys); return () => node.removeEventListener("keydown",keys);
  },[scope,dispatch]);
  return [history.present, value => dispatch({type:"set",value}), {undo:()=>dispatch({type:"undo"}),redo:()=>dispatch({type:"redo"}),canUndo:!!history.past.length,canRedo:!!history.future.length}];
}

import {createContext,useCallback,useContext,useEffect,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {initialRoute,parseWorkspaceRoute,workspaceHref,workspaceTitle} from './workspaceRoutes';

const NavigationContext = createContext(null);
const scrollPosition = () => [window.scrollX,window.scrollY];
const historyState = () => window.history.state && typeof window.history.state === 'object' ? window.history.state : {};
export function WorkspaceNavigation({children}) {
  const [route,setRoute] = useState(() => parseWorkspaceRoute(window.location,window.innerWidth >= 1200));
  const intent = useRef('replace'),first = useRef(true),restoreScroll = useRef(null);
  const navigate = useCallback((patch,{replace=false}={}) => {
    intent.current = replace ? 'replace' : 'push';
    setRoute(previous => {
      const next = typeof patch === 'function' ? patch(previous) : {...previous,...patch};
      return Object.keys({...previous,...next}).every(key => Object.is(previous[key],next[key])) ? previous : next;
    });
  },[]);
  const set = useCallback((name,value) => navigate(previous => ({...previous,[name]:typeof value === 'function' ? value(previous[name]) : value})),[navigate]);
  useEffect(() => {
    const previous = window.history.scrollRestoration;
    window.history.scrollRestoration = 'manual';
    const pop = event => {
      intent.current = 'replace';
      const position = event.state?.classioScroll;
      restoreScroll.current = Array.isArray(position) && position.length === 2 && position.every(value=>Number.isFinite(value) && value>=0 && value<=1e7) ? position : [0,0];
      setRoute(parseWorkspaceRoute(window.location));
    };
    window.addEventListener('popstate',pop);
    return () => {window.removeEventListener('popstate',pop);window.history.scrollRestoration=previous;};
  },[]);
  useLayoutEffect(() => {
    const href = workspaceHref(route),current = window.location.pathname + window.location.search + window.location.hash;
    const replacing = first.current || intent.current === 'replace';
    if(href !== current || first.current) {
      if(!replacing) window.history.replaceState({...historyState(),classioScroll:scrollPosition()},'',window.location.href);
      window.history[replacing?'replaceState':'pushState']({...historyState(),classioRoute:1,...(!replacing?{classioScroll:[0,0]}:{})},'',href);
      if(!replacing) restoreScroll.current=[0,0];
    }
    first.current=false;intent.current='push';document.title=workspaceTitle(route);
    let frame,nextFrame;
    if(restoreScroll.current) {
      const [left,top]=restoreScroll.current;restoreScroll.current=null;
      frame=requestAnimationFrame(()=>{nextFrame=requestAnimationFrame(()=>{
        window.scrollTo({left,top,behavior:'instant'});
        const heading=document.querySelector('.workspace-content h1,.workspace-content h2');
        if(heading){heading.tabIndex=-1;heading.focus({preventScroll:true});}
      });});
    }
    return()=>{cancelAnimationFrame(frame);cancelAnimationFrame(nextFrame);};
  },[route]);
  const value=useMemo(()=>({route,navigate,set}),[route,navigate,set]);
  return <NavigationContext.Provider value={value}>{children}</NavigationContext.Provider>;
}
export function useWorkspaceNavigation() {
  const value=useContext(NavigationContext);
  if(!value) throw new Error('Workspace navigation needs its provider.');
  return value;
}
// Real links preserve native new-tab, middle-click and copy-link behavior.
export function WorkspaceLink({to,href,onClick,children,...props}) {
  const navigation=useContext(NavigationContext);
  const current={...initialRoute,...navigation?.route};
  const target=typeof to==='function'?to(current):{...current,...to};
  const destination=href || workspaceHref(target);
  return <a {...props} href={destination} onClick={event=>{
    if(event.defaultPrevented || event.button!==0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || (event.currentTarget.target && event.currentTarget.target!=='_self')) return;
    if(!onClick && !navigation) return;
    event.preventDefault();
    if(onClick) onClick(event); else navigation.navigate(target);
  }}>{children}</a>;
}

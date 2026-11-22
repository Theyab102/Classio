export const toolPaths = {
  view: 'source', notes: 'notes', quiz: 'quizzes', cards: 'flashcards',
  voice: 'podcasts', present: 'presentations', game: 'games', youtube: 'link',
};
const pathTools = {...Object.fromEntries(Object.entries(toolPaths).map(([key,value]) => [value,key])),recordings:'voice'};
export const initialRoute = {screen:'home',homeTab:'folders',folderId:null,fileId:null,fileTab:'view',groupId:null,assistant:false,audioMode:'podcast'};
const validId = id => typeof id === 'string' && id.length > 0 && id.length <= 300 && !id.includes('/') && !id.includes('\\') && !['.','..'].includes(id) && [...id].every(character=>character.codePointAt(0)>=32 && character.codePointAt(0)!==127);
export function defaultFileTool(file) {
  if(file?.type === 'text/youtube') return 'youtube';
  if(file?._isBlank || file?.isBlankDoc || file?.id?.startsWith('blank_'))
    return toolPaths[file.defaultTab] ? file.defaultTab : 'notes';
  return 'view';
}
export function parseWorkspaceRoute(location, defaultAssistant = false) {
  const url = new URL(typeof location === 'string' ? location : location.href, 'https://classio.invalid');
  const pathname = url.pathname.replace(/\/+$/, '') || '/';
  const base = {...initialRoute,assistant:url.searchParams.get('assistant') === '1' || (pathname === '/' && !url.searchParams.has('assistant') && defaultAssistant)};
  if(pathname === '/' || pathname === '/library') return base;
  if(pathname === '/guides') return {...base,homeTab:'about'};
  if(pathname === '/settings') return {...base,homeTab:'settings'};
  let parts;
  try { parts = pathname.split('/').slice(1).map(decodeURIComponent); } catch { return {...base,screen:'missing',pathname}; }
  if(parts[0] === 'folders' && parts.length === 2 && validId(parts[1]))
    return {...base,screen:'folder',folderId:parts[1]};
  if(parts[0] === 'files' && [2,3].includes(parts.length) && validId(parts[1]) && (!parts[2] || pathTools[parts[2]]))
    return {...base,screen:'file',fileId:parts[1],fileTab:pathTools[parts[2]] || 'view',audioMode:parts[2]==='recordings'?'record':'podcast'};
  if(parts[0] === 'groups' && parts.length === 2 && validId(parts[1]))
    return {...base,screen:'studyGroup',groupId:parts[1]};
  return {...base,screen:'missing',pathname};
}
export function workspaceHref(route) {
  let path = '/library';
  if(route.screen === 'home') path = route.homeTab === 'settings' ? '/settings' : route.homeTab === 'about' ? '/guides' : '/library';
  if(route.screen === 'folder' && validId(route.folderId)) path = '/folders/' + encodeURIComponent(route.folderId);
  if(route.screen === 'file' && validId(route.fileId)) path = '/files/' + encodeURIComponent(route.fileId) + '/' + (route.fileTab==='voice'&&route.audioMode==='record'?'recordings':toolPaths[route.fileTab] || 'source');
  if(route.screen === 'studyGroup' && validId(route.groupId)) path = '/groups/' + encodeURIComponent(route.groupId);
  if(route.screen === 'missing' && route.pathname?.startsWith('/') && !route.pathname.startsWith('//')) path = route.pathname;
  return path + (route.assistant ? '?assistant=1' : '');
}
export function workspaceTitle(route) {
  const tools = {view:'Source',notes:'Notes',quiz:'Quizzes',cards:'Flashcards',voice:'Podcasts',present:'Presentations',game:'Study games',youtube:'Video & links'};
  const title = route.screen === 'file' ? (route.fileTab==='voice'&&route.audioMode==='record'?'Recordings':tools[route.fileTab] || 'Study material') : route.screen === 'folder' ? 'Folder' : route.screen === 'studyGroup' ? 'Study group' : route.screen === 'missing' ? 'Page unavailable' : route.homeTab === 'settings' ? 'Settings' : route.homeTab === 'about' ? 'Study guides' : 'Library';
  return `${title} · Classio`;
}
export const fileDestination = file => ({screen:'file',fileId:file.id,fileTab:defaultFileTool(file),audioMode:file.voiceMode==='record'?'record':'podcast'});

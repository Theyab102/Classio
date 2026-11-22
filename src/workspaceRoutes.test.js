import {defaultFileTool,fileDestination,initialRoute,parseWorkspaceRoute,workspaceHref} from './workspaceRoutes';
test('every main menu and study tool has a reversible same-domain URL',()=>{
 const paths=['/library','/guides','/settings','/folders/physics','/groups/group-one',...['source','notes','quizzes','flashcards','podcasts','recordings','presentations','games','link'].map(tool=>'/files/material/'+tool)];
 for(const path of paths){const route=parseWorkspaceRoute(path);expect(route.screen).not.toBe('missing');expect(workspaceHref(route)).toBe(path);expect(workspaceHref({...route,assistant:true})).toBe(path+'?assistant=1');}
});
test('IDs round-trip without leaking document names, content or full file objects',()=>{
 const id='chapter + ?# العربية';const href=workspaceHref({...initialRoute,screen:'file',fileId:id,fileTab:'notes',name:'PRIVATE TITLE',notes:'PRIVATE CONTENT'});
 expect(parseWorkspaceRoute(href).fileId).toBe(id);expect(href).not.toMatch(/PRIVATE|notes=/);expect(href.startsWith('/files/')).toBe(true);
});
test('malformed paths and unknown tools fail closed instead of opening arbitrary material',()=>{
 for(const path of ['/files/%E0%A4/notes','/files/a/unknown','/files/a/notes/extra','/files/%2f/notes','/folders/%00','/api/study','/.env'])expect(parseWorkspaceRoute(path).screen).toBe('missing');
});
test('blank notes, uploaded sources and video links retain their existing default tools',()=>{
 expect(defaultFileTool({id:'blank_one',_isBlank:true})).toBe('notes');expect(defaultFileTool({_isBlank:true,defaultTab:'voice'})).toBe('voice');
 expect(defaultFileTool({type:'text/youtube'})).toBe('youtube');expect(defaultFileTool({type:'application/pdf'})).toBe('view');
 expect(workspaceHref({...initialRoute,...fileDestination({id:'blank_one',_isBlank:true,defaultTab:'voice'})})).toBe('/files/blank_one/podcasts');
});

import {act,fireEvent,render,screen,waitFor} from '@testing-library/react';
import {useState} from 'react';
import {WorkspaceNavigation,WorkspaceLink,useWorkspaceNavigation} from './WorkspaceNavigation';
const originalScroll=window.scrollTo;
beforeEach(()=>{window.history.replaceState(null,'','/library');window.scrollTo=jest.fn();});
afterEach(()=>{window.scrollTo=originalScroll;window.history.replaceState(null,'','/');});
function Fixture(){const {route,navigate}=useWorkspaceNavigation(),[draft,setDraft]=useState('');return <><WorkspaceLink to={{screen:'home',homeTab:'settings'}}>Settings</WorkspaceLink><WorkspaceLink to={{screen:'file',fileId:'one',fileTab:'notes'}}>Notes</WorkspaceLink><WorkspaceLink to={{screen:'file',fileId:'one',fileTab:'quiz'}}>Quizzes</WorkspaceLink><WorkspaceLink to={current=>({...current,assistant:!current.assistant})}>Assistant</WorkspaceLink><input aria-label="Draft" value={draft} onChange={event=>setDraft(event.target.value)}/><button onClick={()=>navigate({...route},{replace:true})}>Normalize</button><output data-testid="route">{route.screen}:{route.homeTab}:{route.fileTab}:{String(route.assistant)}</output></>;}
test('Back and Forward restore menus without remounting the workspace or losing drafts',async()=>{
 render(<WorkspaceNavigation><Fixture/></WorkspaceNavigation>);const start=window.history.length;
 fireEvent.click(screen.getByRole('link',{name:'Notes'}));expect(window.location.pathname).toBe('/files/one/notes');
 fireEvent.change(screen.getByLabelText('Draft'),{target:{value:'Unsaved form draft'}});expect(window.history.length).toBe(start+1);
 fireEvent.click(screen.getByRole('link',{name:'Quizzes'}));expect(window.location.pathname).toBe('/files/one/quizzes');
 act(()=>window.history.back());await waitFor(()=>expect(window.location.pathname).toBe('/files/one/notes'));await waitFor(()=>expect(screen.getByTestId('route')).toHaveTextContent(':notes:'));expect(screen.getByLabelText('Draft')).toHaveValue('Unsaved form draft');
 act(()=>window.history.forward());await waitFor(()=>expect(screen.getByTestId('route')).toHaveTextContent(':quiz:'));expect(window.location.pathname).toBe('/files/one/quizzes');
});
test('initial direct URLs and canonical replacements do not create duplicate history entries',()=>{
 window.history.replaceState(null,'','/files/one/flashcards?assistant=1');const start=window.history.length;
 render(<WorkspaceNavigation><Fixture/></WorkspaceNavigation>);expect(screen.getByTestId('route')).toHaveTextContent('file:folders:cards:true');expect(window.history.length).toBe(start);
 fireEvent.click(screen.getByText('Normalize'));expect(window.history.length).toBe(start);expect(document.title).toBe('Flashcards · Classio');
 expect(JSON.stringify(window.history.state)).not.toMatch(/Unsaved|Draft|PRIVATE/);
});
test('assistant opening is URL state and browser Back closes it',async()=>{
 render(<WorkspaceNavigation><Fixture/></WorkspaceNavigation>);fireEvent.click(screen.getByRole('link',{name:'Assistant'}));expect(window.location.search).toBe('?assistant=1');
 act(()=>window.history.back());await waitFor(()=>expect(screen.getByTestId('route')).toHaveTextContent(':false'));expect(window.location.search).toBe('');
});
test('modified clicks keep native browser navigation and skip the SPA callback',()=>{
 const select=jest.fn();render(<WorkspaceNavigation><WorkspaceLink to={{screen:'home',homeTab:'settings'}} onClick={select}>Settings</WorkspaceLink></WorkspaceNavigation>);
 const link=screen.getByRole('link',{name:'Settings'});expect(link).toHaveAttribute('href','/settings');
 let prevented;window.addEventListener('click',event=>{prevented=event.defaultPrevented;event.preventDefault();},{once:true});
 fireEvent.click(link,{ctrlKey:true});expect(prevented).toBe(false);expect(select).not.toHaveBeenCalled();expect(window.location.pathname).toBe('/library');
 fireEvent.click(link);expect(select).toHaveBeenCalledTimes(1);
});

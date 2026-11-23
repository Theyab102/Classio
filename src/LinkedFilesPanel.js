import {useState} from 'react';
import StudioDialog from './StudioDialog';
import UiIcon from './UiIcon';
import {WorkspaceLink} from './WorkspaceNavigation';
import {fileDestination} from './workspaceRoutes';
import {linkedFilesFor} from './linkedFiles';

export default function LinkedFilesPanel({file,files,onSave,compact=false}) {
  const [open,setOpen]=useState(false),[query,setQuery]=useState('');
  const linked=linkedFilesFor(file,files),others=files.filter(item=>item.id!==file.id);
  const selected=new Set(linked.map(item=>item.id));
  return <section className={`linked-materials ${compact?'is-compact':''}`} aria-label={`Linked sources for ${file.name}`}>
    <div className="linked-materials-heading"><button className="quiet-button" type="button" onClick={()=>setOpen(true)}><UiIcon name="link" size={16}/>{linked.length?`${linked.length} linked ${linked.length===1?'source':'sources'}`:'Link files'}</button>{!compact&&<p>Use a revision pack and its answer key together in AI study tools.</p>}</div>
    {!compact&&!!linked.length&&<ul>{linked.map(item=><li key={item.id}><WorkspaceLink to={fileDestination(item)}><UiIcon name="source" size={15}/>{item.name}</WorkspaceLink></li>)}</ul>}
    {open&&<StudioDialog title="Link study files" onClose={()=>setOpen(false)}><div className="simple-workspace-form"><p>Link sources in either direction. Notes, questions and the assistant can read both files. Existing notes stay as they are until you generate again.</p><label>Find a file<input type="search" value={query} onChange={event=>setQuery(event.target.value)} placeholder="Revision pack, answer key…"/></label><div className="linked-materials-options">{others.filter(item=>`${item.name} ${item.folderName||''}`.toLowerCase().includes(query.toLowerCase())).map(item=><label className="linked-material-option" key={item.id}><input type="checkbox" checked={selected.has(item.id)} onChange={event=>onSave(event.target.checked?[...selected,item.id]:[...selected].filter(id=>id!==item.id))}/><span><strong>{item.name}</strong>{item.folderName&&<small>{item.folderName}</small>}</span></label>)}{!others.length&&<p>Add another file to your Library to link it here.</p>}</div><button className="quiet-button is-primary" onClick={()=>setOpen(false)}>Done</button></div></StudioDialog>}
  </section>;
}

import {useState} from 'react';
import StudioDialog from './StudioDialog';
import UiIcon from './UiIcon';

export default function CloudSyncNotice({message,onRetry}) {
  const retrying=message==='Retrying cloud sync…';
  const localFailure=/Browser storage.*(?:full|unavailable)|could not be saved|export important work/.test(message);
  const summary=retrying?'Syncing…':localFailure?'Saving needs attention — keep this tab open.':/Original uploads could not be restored/.test(message)?'Original uploads could not be restored':/Cloud sync is unavailable/.test(message)?'Cloud sync is unavailable':/sync|Cloud access|Cloud sign-in/.test(message)?'Cloud sync needs attention':message;
  return <div className="workspace-warning sync-status" aria-busy={retrying}><UiIcon name="rewind" size={16}/><p role="status" title={message}>{summary}</p>{onRetry&&<button className="quiet-button" aria-label="Retry sync" disabled={retrying} onClick={onRetry}>Retry</button>}</div>;
}

export function CloudSyncSettings({message,diagnostic}) {
  const [open,setOpen]=useState(false);
  return <div className="cloud-sync-settings"><p>{message||'Cloud sync is enabled for this account.'}</p>
    {diagnostic&&<button type="button" className="quiet-button" onClick={()=>setOpen(true)}>Sync troubleshooting</button>}
    {open&&<StudioDialog title="Sync troubleshooting" onClose={()=>setOpen(false)}><div className="cloud-sync-details"><p>For the site owner or support. These details contain no credentials, account IDs or document contents.</p><dl>
      {Object.entries({Project:diagnostic.project,Operation:diagnostic.operation||'account sync',Document:diagnostic.path||'users/{uid}',Error:diagnostic.code,'Signed in':diagnostic.signedIn?'Yes':'No','Sign-in provider':diagnostic.provider,'Provider check':diagnostic.providerEvidence}).map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
    </dl><p>The site owner must check the deployed Firebase rules for these private account paths. Retrying cannot change those permissions.</p></div></StudioDialog>}
  </div>;
}

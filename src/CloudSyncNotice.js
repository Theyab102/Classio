export default function CloudSyncNotice({message,diagnostic,onRetry}) {
  return <div className="workspace-warning"><p role="status">{message}</p>{onRetry&&<button className="quiet-button" disabled={message==='Retrying cloud sync…'} onClick={onRetry}>Retry sync</button>}
    {diagnostic&&<details className="cloud-sync-details"><summary>Cloud sync details for the site owner</summary><dl>
      {Object.entries({Project:diagnostic.project,Operation:diagnostic.operation||'account sync',Document:diagnostic.path||'users/{uid}',Error:diagnostic.code,'Signed in':diagnostic.signedIn?'Yes':'No','Sign-in provider':diagnostic.provider,'Provider check':diagnostic.providerEvidence}).map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
    </dl><p>These details contain no credentials or document contents. The owner must compare the deployed Firestore rules with these private account paths.</p></details>}
  </div>;
}

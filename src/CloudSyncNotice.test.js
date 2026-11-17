import {fireEvent,render,screen} from '@testing-library/react';
import CloudSyncNotice,{CloudSyncSettings} from './CloudSyncNotice';
const diagnostic={project:'classio-4378f',operation:'read manifest',path:'users/{uid}/preferences/workspace-v2',code:'permission-denied',signedIn:true,provider:'google.com',providerEvidence:'refreshed ID token'};
test('the ordinary sync status stays compact and does not expose owner troubleshooting fields',()=>{
 const retry=jest.fn();render(<CloudSyncNotice message="Cloud access was denied. Sign in again; check the deployed Firebase rules." diagnostic={diagnostic} onRetry={retry}/>);
 expect(screen.getByRole('status')).toHaveTextContent('Cloud sync needs attention');expect(screen.queryByText('classio-4378f')).not.toBeInTheDocument();expect(screen.queryByRole('dialog')).not.toBeInTheDocument();fireEvent.click(screen.getByRole('button',{name:'Retry sync'}));expect(retry).toHaveBeenCalledTimes(1);
});
test('technical diagnostics require an intentional Settings action and stay out of the page layout',()=>{
 render(<CloudSyncSettings message="Cloud access was denied." diagnostic={diagnostic}/>);expect(screen.queryByText('classio-4378f')).not.toBeInTheDocument();fireEvent.click(screen.getByRole('button',{name:'Sync troubleshooting'}));expect(screen.getByRole('dialog',{name:'Sync troubleshooting'})).toBeInTheDocument();expect(screen.getByText('users/{uid}/preferences/workspace-v2')).toBeInTheDocument();fireEvent.click(screen.getByRole('button',{name:'Close Sync troubleshooting'}));expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});
test('compacting cloud errors preserves local file deletion instructions',()=>{
 const message='This browser could not finish deleting local file/audio copies. Clear this site’s storage on this device to remove them.';
 render(<CloudSyncNotice message={message}/>);expect(screen.getByRole('status')).toHaveTextContent(message);
});

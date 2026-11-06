import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {useState} from 'react';
import AssistantAction from './AssistantActions';
import {requestJson} from './apiClient';
jest.mock('./firebase',()=>({auth:{currentUser:{uid:'alice'}},accountHeaders:jest.fn(async()=>({Authorization:'Bearer test-only-token'}))}));
jest.mock('./apiClient',()=>({requestJson:jest.fn()}));
const action={tool:'create_notes',requestId:'12345678-1234-1234-1234-123456789abc',arguments:{fileId:'source',title:'My energy notes',difficulty:'beginner',count:2,instructions:'Explain energy.'}};
beforeEach(()=>{requestJson.mockReset();});
test('study actions require an explicit click and acknowledge saving only after confirmation from the backend',async()=>{
  const open=jest.fn();requestJson.mockResolvedValue({saved:true,file:{id:'result',folderId:'school',name:'My energy notes',defaultTab:'notes'}});
  function Harness(){const [value,setValue]=useState(action);return <AssistantAction action={value} onComplete={result=>setValue({...value,result})} onOpenFile={open}/>;}
  render(<Harness/>);expect(requestJson).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button',{name:'Confirm and create'}));
  await screen.findByText('Saved to your account.');
  expect(requestJson.mock.calls[0][1].body).toMatchObject({tool:'create_notes',confirmed:true,requestId:action.requestId});
  fireEvent.click(screen.getByRole('button',{name:'Open saved file'}));await waitFor(()=>expect(open).toHaveBeenCalledWith(expect.objectContaining({fileId:'result'})));
});
test('an unconfirmed save leaves retry available and never announces success',async()=>{
  requestJson.mockResolvedValue({saved:false});const complete=jest.fn();
  render(<AssistantAction action={action} onComplete={complete}/>);fireEvent.click(screen.getByRole('button',{name:'Confirm and create'}));
  await screen.findByRole('alert');expect(complete).not.toHaveBeenCalled();expect(screen.queryByText('Saved to your account.')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'Retry action'}));await waitFor(()=>expect(requestJson).toHaveBeenCalledTimes(2));
  expect(requestJson.mock.calls[1][1].body.requestId).toBe(action.requestId);
});

import {watchVerificationSync} from './verificationSync';
import {auth,linkVerification} from './firebase';
jest.mock('./firebase',()=>({auth:{currentUser:{uid:'alice'}},linkVerification:jest.fn()}));
const flush=()=>new Promise(jest.requireActual('timers').setImmediate);
beforeEach(()=>{jest.useFakeTimers();auth.currentUser={uid:'alice'};linkVerification.mockReset();});
afterEach(()=>jest.useRealTimers());
test('account verification retries a temporary failure without blocking the caller',async()=>{
  const status=jest.fn();linkVerification.mockRejectedValueOnce(new Error('offline')).mockResolvedValue(undefined);
  const stop=watchVerificationSync('alice',status);expect(typeof stop).toBe('function');
  try{await flush();jest.advanceTimersByTime(5000);await flush();
  expect(linkVerification).toHaveBeenCalledTimes(2);expect(status).toHaveBeenCalledWith('synced');}finally{stop();}
});
test('sign-out cancels a pending retry and cannot sync the previous user from another account',async()=>{
  linkVerification.mockRejectedValue(new Error('offline'));const status=jest.fn(),stop=watchVerificationSync('alice',status);
  try{await flush();auth.currentUser={uid:'bob'};jest.advanceTimersByTime(5000);await flush();
  expect(linkVerification).toHaveBeenCalledTimes(1);expect(status).not.toHaveBeenCalled();}finally{stop();}
});

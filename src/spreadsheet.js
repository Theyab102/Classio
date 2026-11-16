import {verifyDocumentArchive} from './documentArchive';
export async function readSpreadsheet(file){
 if(file.size>20*1024*1024)throw new Error('Choose a spreadsheet under 20 MB.');
 if(/\.xlsx$/i.test(file.name||''))await verifyDocumentArchive(file);
 const bytes=await file.arrayBuffer();
 return new Promise((resolve,reject)=>{
  const worker=new Worker('/spreadsheet-worker.js'),finish=(error,result)=>{clearTimeout(timer);worker.terminate();error?reject(error):resolve(result);};
  const timer=setTimeout(()=>finish(new Error('This spreadsheet took too long to read. Open the original in your spreadsheet app.')),8000);
  worker.onmessage=event=>event.data.error?finish(new Error(event.data.error)):finish(null,event.data.sheets);
  worker.onerror=()=>finish(new Error('The spreadsheet reader could not start. Download the original file.'));
  worker.postMessage(bytes,[bytes]);
 });
}

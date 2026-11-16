// Isolate the legacy XLS parser from DOM/account data and bound execution time
// from the caller. The deployment gives this worker a separate restrictive CSP.
importScripts('/vendor/xlsx.min.js');
self.onmessage=event=>{
 try{
  const book=self.XLSX.read(event.data,{type:'array',sheetRows:2000,bookVBA:false,cellHTML:false,cellFormula:false});
  const sheets=book.SheetNames.slice(0,20).map(name=>({name,rows:self.XLSX.utils.sheet_to_json(book.Sheets[name],{header:1,defval:'',raw:false}).slice(0,2000).map(row=>row.slice(0,100).map(value=>String(value??'').slice(0,5000)))}));
  self.postMessage({sheets});
 }catch{self.postMessage({error:'This spreadsheet could not be read. Download the original to open it in your spreadsheet app.'});}
};

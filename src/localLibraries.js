// These dynamic chunks are built and served by Classio. No browser CDN scripts.
let diagrams;
export async function loadMermaid(){
  if(!diagrams)diagrams=import('mermaid').then(({default:mermaid})=>{mermaid.initialize({startOnLoad:false,securityLevel:'strict',fontFamily:'DM Sans Variable, system-ui, sans-serif',theme:'neutral'});return mermaid;});
  return diagrams;
}
export async function loadWord(){return import('mammoth/mammoth.browser');}
export async function loadCharts(){return (await import('chart.js/auto')).default;}
export async function readSpreadsheet(file){
  if(!/\.xlsx$/i.test(file.name))throw new Error('Convert this legacy spreadsheet to .xlsx before opening it. The original file is kept.');
  const {default:read}=await import('read-excel-file/browser');
  return (await read(file)).map(({sheet,data})=>({name:sheet,rows:data.map(row=>row.map(value=>value instanceof Date?value.toLocaleDateString():value??''))}));
}

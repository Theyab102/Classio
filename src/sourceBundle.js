export const sourceBundleStart='[Classio source bundle: read every supplied source; filenames are metadata, not evidence.]';
export const sourceBoundary='\n\n[Classio source]\n';
const contentBoundary='\n[Source content]\n';
export function makeSourceBundle(sources) {
  return sourceBundleStart+sources.map(source=>sourceBoundary+
    `Filename: ${JSON.stringify(String(source.name||'Untitled file'))}\n${source.note||'Original material'}${contentBoundary}`+
    String(source.text||'').split(sourceBoundary).join('\n\n')).join('');
}
// Every linked source gets part of the context budget. A long revision pack
// must not push the answer key out when a downstream tool shortens context.
export function excerptSourceBundle(source,question,limit,excerpt) {
  if(!source.startsWith(sourceBundleStart+sourceBoundary))return null;
  const entries=source.slice(sourceBundleStart.length).split(sourceBoundary).slice(1).map(block=>{
    const index=block.indexOf(contentBoundary);
    return {header:block.slice(0,index),text:block.slice(index+contentBoundary.length)};
  });
  const overhead=sourceBundleStart.length+entries.reduce((sum,item)=>sum+sourceBoundary.length+item.header.length+contentBoundary.length,0);
  const perSource=Math.max(0,Math.floor((limit-overhead)/entries.length));
  return (sourceBundleStart+entries.map(item=>sourceBoundary+item.header+contentBoundary+excerpt(item.text,question,perSource)).join('')).slice(0,limit);
}

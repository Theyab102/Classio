// Optional licensed search adapters; credentials stay on the server.
function searchProviders(env=process.env) {
  return [env.PEXELS_API_KEY && 'pexels', env.UNSPLASH_ACCESS_KEY && 'unsplash', env.GOOGLE_IMAGE_SEARCH_KEY && env.GOOGLE_IMAGE_SEARCH_CX && 'google'].filter(Boolean);
}
function https(value,hosts) { try {const u=new URL(value);return u.protocol==='https:'&&(!hosts||hosts.some(host=>u.hostname===host||u.hostname.endsWith('.'+host)))?u.href:'';}catch{return '';} }
const clean=value=>String(value||'').replace(/<[^>]*>/g,'').slice(0,200);
async function searchImages(query,{env=process.env,fetcher=global.fetch}={}) {
  const term=typeof query==='string'?query.trim().slice(0,100):'';
  if(!term)return {images:[],unavailable:[]};
  const providers=searchProviders(env),unavailable=[];
  async function json(url,headers){const response=await fetcher(url,{headers,signal:AbortSignal.timeout(7000)});if(!response.ok)throw new Error('Search unavailable');return response.json();}
  const groups=await Promise.all(providers.map(async provider=>{try{
    if(provider==='pexels') {const data=await json('https://api.pexels.com/v1/search?'+new URLSearchParams({query:term,per_page:'6'}),{Authorization:env.PEXELS_API_KEY});return(data.photos||[]).map(p=>({src:https(p.src?.large,['images.pexels.com']),source:https(p.url,['pexels.com']),title:clean(p.alt||term),credit:clean(p.photographer)+' · Pexels',width:p.width,height:p.height}));}
    if(provider==='unsplash') {const data=await json('https://api.unsplash.com/search/photos?'+new URLSearchParams({query:term,per_page:'6'}),{Authorization:'Client-ID '+env.UNSPLASH_ACCESS_KEY,'Accept-Version':'v1'});return(data.results||[]).map(p=>({src:https(p.urls?.regular,['images.unsplash.com']),source:https(p.links?.html,['unsplash.com']),title:clean(p.alt_description||p.description||term),credit:clean(p.user?.name)+' · Unsplash',creator:https(p.user?.links?.html,['unsplash.com']),provider:'unsplash',providerId:/^[\w-]{1,80}$/.test(p.id||'')?p.id:'',width:p.width,height:p.height}));}
    const data=await json('https://customsearch.googleapis.com/customsearch/v1?'+new URLSearchParams({key:env.GOOGLE_IMAGE_SEARCH_KEY,cx:env.GOOGLE_IMAGE_SEARCH_CX,q:term,searchType:'image',rights:'cc_publicdomain|cc_attribute|cc_sharealike',num:'6'}));
    // Require explicit license metadata as well as Google's rights filter.
    return(data.items||[]).flatMap(p=>{const license=(p.pagemap?.metatags||[]).flatMap(meta=>[meta['license'],meta['copyright'],meta['dc.rights']]).find(value=>/creativecommons.org\/(?:licenses|publicdomain)\//i.test(value||''));if(!license)return[];return[{src:https(p.link),source:https(p.image?.contextLink),title:clean(p.title),credit:'Source via Google · '+clean(license),width:p.image?.width,height:p.image?.height}];});
  }catch{unavailable.push(provider);return[];}}));
  return {images:groups.flat().filter(image=>image.src&&image.source).slice(0,18),unavailable};
}
async function trackImageDownload(id,{env=process.env,fetcher=global.fetch}={}) {
  if(!env.UNSPLASH_ACCESS_KEY||!/^[\w-]{1,80}$/.test(id||''))return false;
  const response=await fetcher('https://api.unsplash.com/photos/'+encodeURIComponent(id)+'/download',{headers:{Authorization:'Client-ID '+env.UNSPLASH_ACCESS_KEY,'Accept-Version':'v1'},signal:AbortSignal.timeout(7000)});return response.ok;
}
module.exports={searchProviders,searchImages,trackImageDownload};

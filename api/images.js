const { reserveRequest }=require("../server/ai-quota");
const { imageCapabilities,imageOperation }=require("../server/image-service");
const {searchProviders,searchImages,trackImageDownload}=require('../server/image-search');
module.exports=async function images(req,res){
  res.setHeader("Cache-Control","no-store");
  if(req.method==='GET')return res.status(200).json({...imageCapabilities(),searchProviders:searchProviders()});
  if(req.method!=='POST'){res.setHeader("Allow","GET, POST");return res.status(405).json({error:"Use POST."});}
  if((req.headers['sec-fetch-site'] && req.headers['sec-fetch-site']!=='same-origin') || !/^application\/json(?:\s*;|$)/i.test(req.headers['content-type']||''))return res.status(403).json({error:"Open Classio to edit images."});
  try{const quota=await reserveRequest(req,req.body?.action==="search"?"image-search":req.body?.action==="track-download"?"image-attribution":"image");if(quota.unavailable)return res.status(503).json({error:"Image requests are unavailable."});if(quota.retryAfter){res.setHeader("Retry-After",String(quota.retryAfter));return res.status(429).json({error:"The image request limit was reached. Try again later."});}if(req.body?.action==='search')return res.status(200).json(await searchImages(req.body.query));if(req.body?.action==='track-download')return res.status(200).json({tracked:await trackImageDownload(req.body.id)});return res.status(200).json(await imageOperation(req.body||{}));}
  catch(error){return res.status(error.status||503).json({error:error.status?error.message:"The image service is unavailable. Your original image is kept."});}
};

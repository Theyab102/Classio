const {createHash,createHmac}=require('node:crypto');
const {jsonRequest,reject,publicError}=require('../server/http');
const {requireAccess}=require('../server/access');
const {configured,reserveRequest}=require('../server/ai-quota');
const {podcastInput,podcastFingerprint}=require('../src/shared/podcast-schema');

module.exports=async(req,res)=>{
  if(!jsonRequest(req,res))return;
  const input=podcastInput(req.body);
  if(!input)return reject(res,400,'INVALID_PODCAST','Check the dialogue, English language, and distinct voice selections.');
  try{
    const secret=process.env.CLASSIO_TTS_SIGNING_SECRET;
    if(!configured() || (secret||'').length<32)return reject(res,503,'PODCAST_CONFIG_MISSING','Podcast audio needs the voice server and its shared signing secret configured. Your script is kept.');
    const access=await requireAccess(req);
    const quota=await reserveRequest(req,'podcast',access.user?.uid);
    if(quota.unavailable)return reject(res,503,'QUOTA_UNAVAILABLE','Audio usage limits could not be checked. Please retry.');
    if(quota.retryAfter){res.setHeader('Retry-After',String(quota.retryAfter));return reject(res,429,'USAGE_LIMIT','The podcast audio limit was reached. Your script is kept; retry later.');}
    const owner=createHmac('sha256',secret).update(access.user?.uid || req.headers.cookie || '').digest('hex');
    const hash=createHash('sha256').update(podcastFingerprint(input)).digest('hex');
    const exp=Math.floor(Date.now()/1000)+600;
    const body=Buffer.from(JSON.stringify({o:owner,h:hash,exp})).toString('base64url');
    const signature=createHmac('sha256',secret).update(body).digest('base64url');
    res.status(200).json({ticket:body+'.'+signature,expiresAt:exp*1000});
  }catch(error){return reject(res,publicError(error)?error.status:503,publicError(error)?error.code:'PODCAST_UNAVAILABLE',publicError(error)?error.message:'Podcast authorization could not finish. Your script is kept.');}
};

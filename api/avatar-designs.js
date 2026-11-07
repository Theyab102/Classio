const {cert,getApps,initializeApp}=require('firebase-admin/app');
const {getAuth}=require('firebase-admin/auth');
const {getFirestore,FieldValue}=require('firebase-admin/firestore');
const {reserveRequest}=require('../server/ai-quota');
const {sharedDesign}=require('../server/avatar-sharing');
module.exports=async function avatarDesigns(req,res){
  res.setHeader('Cache-Control','no-store');
  if(!['GET','POST'].includes(req.method)){res.setHeader('Allow','GET, POST');return res.status(405).json({error:'Use GET or POST.'});}
  if(process.env.AVATAR_SHARING_ENABLED!=='true'||!process.env.FIREBASE_SERVICE_ACCOUNT_JSON)return res.status(503).json({error:'Public designs require the site’s moderation service and Firebase admin configuration. Your designs remain private.'});
  try{
    const app=getApps().find(a=>a.name==='classio-api')||initializeApp({credential:cert(JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON))},'classio-api');
    const db=getFirestore(app),designs=db.collection('avatarDesigns');
    if(req.method==='GET'){const rows=await designs.where('status','==','approved').limit(24).get();return res.status(200).json({designs:rows.docs.map(doc=>({id:doc.id,...sharedDesign(doc.data().design)}))});}
    if((req.headers['sec-fetch-site']&&req.headers['sec-fetch-site']!=='same-origin')||!/^application\/json(?:\s*;|$)/i.test(req.headers['content-type']||''))return res.status(403).json({error:'Open Classio to share designs.'});
    const token=/^Bearer (.+)$/.exec(req.headers.authorization||'');if(!token)return res.status(401).json({error:'Sign in to share or report designs.'});
    let user;try{user=await getAuth(app).verifyIdToken(token[1],true);}catch{return res.status(401).json({error:'Sign in again to continue.'});}
    if(user.firebase?.sign_in_provider!=='google.com')return res.status(403).json({error:'Use your verified Classio account to share designs.'});
    const quota=await reserveRequest(req,'design-sharing');if(quota.unavailable||quota.retryAfter)return res.status(429).json({error:'Please wait before submitting another design or report.'});
    if(req.body?.action==='submit'){const design=sharedDesign(req.body.design);if(!design)return res.status(400).json({error:'Preview and validate the design before submitting it.'});const doc=await designs.add({owner:user.uid,design,status:'pending',createdAt:FieldValue.serverTimestamp()});return res.status(202).json({id:doc.id,status:'pending'});}
    if(req.body?.action==='report'&&/^[\w-]{1,80}$/.test(req.body.id||'')&&typeof req.body.reason==='string'&&req.body.reason.trim().length>=5&&req.body.reason.length<=500){const doc=await designs.doc(req.body.id).get();if(!doc.exists||doc.data().status!=='approved')return res.status(404).json({error:'That public design is unavailable.'});await db.collection('avatarDesignReports').add({designId:doc.id,reporter:user.uid,reason:req.body.reason.trim(),status:'open',createdAt:FieldValue.serverTimestamp()});return res.status(202).json({status:'reported'});}
    return res.status(400).json({error:'Choose a design action.'});
  }catch{return res.status(503).json({error:'The design sharing service is unavailable. Private designs are still available.'});}
};

const {test}=require('node:test'),assert=require('node:assert/strict');
const {configurationStatus,publicStatus}=require('./ai-configuration');
const {reserveInstance,configured,quotaMode}=require('./ai-quota');
const env={Groq_Key:'fixture-provider',AI_GROQ_STUDENT_USE_APPROVED:'true',AI_GROQ_FREE_TIER_CONFIRMED:'true',CLASSIO_SESSION_SECRET:'fixture-session-secret-at-least-32-characters'};
test('local configuration uses exact aliases, signed sessions and free/student eligibility; public status omits credential internals',()=>{
 const report=configurationStatus({...env,Gemini_Key:'fixture-google',Gemini_Key_packup:'fixture-google-backup'});assert(report.configured);assert.deepEqual(report.providers,['groq']);assert.equal(report.providerConfiguration.gemini.reason,'AUDIENCE_RESTRICTED');assert(!JSON.stringify(report).includes('fixture-'));
 const visible=JSON.stringify(publicStatus(env));for(const name of ['Groq_Key','primary','backup','slot','CLASSIO_SESSION_SECRET','missing','settingNames'])assert(!visible.includes(name));
});
test('missing backups are optional; public variables, placeholders, unapproved providers and Gemini alone cannot activate inference',()=>{
 for(const settings of [{REACT_APP_GROQ_KEY:'fixture'},{Groq_Key:'Your_new_Groq_API_key'},{Groq_Key:'gsk_example...'},{Gemini_Key:'fixture',GEMINI_ENABLED:'true'},{...env,CLASSIO_SESSION_SECRET:''},{...env,AI_GROQ_STUDENT_USE_APPROVED:''}])assert(!configurationStatus(settings).configured);
 assert(configurationStatus(env).configured);
});
test('production quota requires shared persistent storage and fails closed for instance mode or invalid admin credentials',()=>{
 assert.equal(quotaMode({VERCEL:'1'}),'firestore');assert(!configured({VERCEL:'1',AI_QUOTA_MODE:'instance'}));assert(!configured({AI_QUOTA_MODE:'firestore'}));assert(!configured({AI_QUOTA_MODE:'bad'}));
 assert(!configured({VERCEL:'1',FIREBASE_SERVICE_ACCOUNT_JSON:JSON.stringify({project_id:'fixture',client_email:'fixture@example.com',private_key:'fixture-private-key'}),CLASSIO_SESSION_SECRET:env.CLASSIO_SESSION_SECRET}));
 const {privateKey}=require('node:crypto').generateKeyPairSync('rsa',{modulusLength:2048});
 const account={project_id:'classio-4378f',client_email:'fixture@example.com',private_key:privateKey.export({format:'pem',type:'pkcs8'}).toString()};
 assert(configured({VERCEL:'1',FIREBASE_SERVICE_ACCOUNT_JSON:JSON.stringify(account),CLASSIO_SESSION_SECRET:env.CLASSIO_SESSION_SECRET}));
 assert(!configured({VERCEL:'1',FIREBASE_SERVICE_ACCOUNT_JSON:JSON.stringify({...account,private_key:'invalid'}),CLASSIO_SESSION_SECRET:env.CLASSIO_SESSION_SECRET}));
});
test('instance limiter enforces per-IP and per-account limits plus shared daily cap, expiring old counters',()=>{
 const store=new Map();for(let i=0;i<40;i++)assert.deepEqual(reserveInstance('study',{ip:'a',user:'account'},1,store),{});
 assert(reserveInstance('study',{ip:'b',user:'account'},1,store).retryAfter);assert.deepEqual(reserveInstance('study','a',600001,store),{});
 for(let i=0;i<209;i++)assert.deepEqual(reserveInstance('study','other-'+i,600001,store),{});assert(reserveInstance('study','final',600001,store).retryAfter);assert.deepEqual(reserveInstance('study','final',86400001,store),{});
});
test('audio/session quotas are separate and cannot be bypassed by guest identity churn on one IP',()=>{
 const store=new Map();for(let i=0;i<3;i++)assert.deepEqual(reserveInstance('audio',{ip:'same',user:'guest-'+i},1,store),{});assert(reserveInstance('audio',{ip:'same',user:'different'},1,store).retryAfter);assert.deepEqual(reserveInstance('study','same',1,store),{});
 for(let i=0;i<8;i++)assert.deepEqual(reserveInstance('session','same',1,store),{});assert(reserveInstance('session','same',1,store).retryAfter);
});

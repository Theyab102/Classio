// Synthetic credentials exist only in an external temporary directory.
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),cp=require('node:child_process'),crypto=require('node:crypto');
const names=['Groq_Key','Groq_Key_packup','Gemini_Key','Gemini_Key_packup','OPENROUTER_API_KEY','OPENROUTER_API_KEY_packup','CLOUDFLARE_API_TOKEN','CLOUDFLARE_API_TOKEN_packup','OLLAMA_API_TOKEN','OLLAMA_API_TOKEN_packup','CLASSIO_SESSION_SECRET'];
const root=fs.mkdtempSync(path.join(os.tmpdir(),'classio-scanner-canaries-')),report=path.join(root,'report.json');
try{
 fs.writeFileSync(path.join(root,'canaries.env'),names.map(name=>name+'='+crypto.randomBytes(36).toString('base64url')).join('\n'));
 const scanned=cp.spawnSync(process.env.GITLEAKS_BIN||'gitleaks',['dir',root,'--config',path.resolve(__dirname,'../.gitleaks.toml'),'--redact=100','--no-banner','--log-level=error','--report-format=json','--report-path='+report,'--exit-code=10'],{encoding:'utf8'});
 if(scanned.error||scanned.status!==10)throw Error('Scanner did not block synthetic private credentials.');
 const rows=JSON.parse(fs.readFileSync(report,'utf8'));
 for(const name of names)if(!rows.some(row=>row.Match.startsWith(name+'=')))throw Error('Scanner missed a synthetic '+name+' assignment.');
 console.log('PASS: scanner blocked all '+names.length+' primary/packup/session canaries; no values printed.');
}finally{fs.rmSync(root,{recursive:true,force:true});}

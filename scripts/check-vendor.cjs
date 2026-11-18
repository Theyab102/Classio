const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'..'),manifest=require('../src/localScriptManifest.json');
for(const asset of manifest){const bytes=fs.readFileSync(path.join(root,'public/vendor',asset.file));if(crypto.createHash('sha384').update(bytes).digest('base64')!==asset.sha384)throw new Error('Local document asset integrity check failed: '+asset.name);}

require('./check-security-vendor.cjs');

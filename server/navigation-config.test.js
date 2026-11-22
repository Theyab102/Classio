const {test}=require('node:test'),assert=require('node:assert/strict');
const config=require('../vercel.json');
const {frontendAsset}=require('../scripts/browser-assets.cjs');
test('refreshable workspace routes preserve API, asset and private-file routing',()=>{
 const spa=config.rewrites.filter(route=>route.destination==='/index.html').map(route=>route.source);
 for(const path of ['/library','/guides','/settings','/folders/:path*','/files/:path*','/groups/:path*'])assert(spa.includes(path));
 assert(!spa.includes('/:path*')&&!spa.includes('/(.*)'));assert(spa.every(path=>!path.startsWith('/api')&&!path.startsWith('/static')&&!path.startsWith('/.')));
 for(const path of ['/.git','/.git/:path*','/.env','/.env.:path*'])assert.equal(config.rewrites.find(route=>route.source===path).destination,'/api/private');
 for(const path of ['/privacy','/terms','/copyright'])assert(spa.includes(path));
});
test('browser fixtures mirror SPA refresh routing without falling back for API or private paths',()=>{
 for(const route of ['/library','/settings','/folders/physics','/files/one/notes','/files/one/podcasts','/groups/study'])assert.equal(frontendAsset('/test/build',route),'/test/build/index.html');
 for(const route of ['/api/study','/.env','/.git/config','/static/js/missing.js'])assert.notEqual(frontendAsset('/test/build',route),'/test/build/index.html');
 assert.throws(()=>frontendAsset('/test/build','/%2e%2e/.env'));
});

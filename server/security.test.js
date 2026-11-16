const {test}=require('node:test'),assert=require('node:assert/strict');
const {safetyResponse}=require('./study-safety');
const config=require('../vercel.json');
test('explicit self-harm disclosure gets support without a provider; ordinary study language does not',()=>{
 assert.match(safetyResponse({messages:[{role:'user',content:'I want to kill myself'}]}),/local emergency/);
 assert.equal(safetyResponse({messages:[{role:'user',content:'This math is killing me. Explain it.'}]}),null);
 assert.equal(safetyResponse({audio:'bytes'}),null);
 assert.equal(safetyResponse({messages:[{role:'user',content:'Explain the phrase I want to kill myself in this book.'}]}),null);
});
test('production deployment constrains scripts, embedding and private checkout paths',()=>{
 const headers=config.headers.find(rule=>rule.source==='/(.*)').headers;
 assert(headers.some(h=>h.key==='Content-Security-Policy'&&h.value.includes("script-src 'self'")&&h.value.includes("frame-ancestors 'none'")));
 assert(config.rewrites.some(rule=>rule.source==='/.git/:path*'&&rule.destination==='/api/private'));
});

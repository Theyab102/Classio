// CRA remains the frontend toolchain. The local API mirrors Vercel handlers.
const http = require('node:http');
const { spawn } = require('node:child_process');
const { randomBytes } = require('node:crypto');
const path = require('node:path');

process.env.NODE_ENV = 'development';
process.env.HOST ||= '127.0.0.1';
process.env.CLASSIO_SESSION_SECRET ||= randomBytes(32).toString('hex');
const routes = new Set(['study', 'access', 'agent', 'podcast']);
const api = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const route = url.pathname.replace(/^\/api\//, '');
  res.status = status => { res.statusCode = status; return res; };
  res.json = data => { res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(data)); };
  if (!routes.has(route)) return res.status(404).json({ error: 'API route not found.' });
  let body = '';
  try {
    for await (const chunk of req) {
      body += chunk;
      if (Buffer.byteLength(body) > 4100000) return res.status(413).json({ error: 'Request is too large.' });
    }
    req.body = body ? JSON.parse(body) : {};
  } catch { return res.status(400).json({ error: 'Invalid JSON request.' }); }
  try { await require(path.resolve(__dirname, '../api', route + '.js'))(req, res); }
  catch { if (!res.writableEnded) res.status(503).json({ error: 'The local API is unavailable. Check the server configuration.' }); }
});
api.listen(Number(process.env.CLASSIO_API_PORT || 3001), '127.0.0.1', () => {
  console.log('Classio local API started. AI still requires real provider and Firebase Admin configuration.');
});
const frontend = spawn(process.execPath, ['--require', path.resolve(__dirname, 'dev-host.cjs'), require.resolve('react-scripts/scripts/start')], { stdio: 'inherit', env: process.env });
function shutdown(signal) { frontend.kill(signal); api.close(); }
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => shutdown(signal));
frontend.on('exit', code => { api.close(); process.exitCode = code || 0; });

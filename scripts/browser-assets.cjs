const path=require('node:path');
const spaSources=require('../vercel.json').rewrites.filter(route=>route.destination==='/index.html').map(route=>route.source);
// Match only the SPA routes declared for production, never API/private/assets.
function frontendAsset(root,pathname) {
  const route=pathname.replace(/\/+$/,'')||'/';
  const app=route==='/'||spaSources.some(source=>source.endsWith('/:path*')
    ? route===source.split('/:path*')[0]||route.startsWith(source.split('/:path*')[0]+'/')
    : route===source);
  const file=path.resolve(root,'.'+(app?'/index.html':decodeURIComponent(pathname)));
  if(!file.startsWith(path.resolve(root)+path.sep))throw new Error('Invalid static path');
  return file;
}
module.exports={frontendAsset};

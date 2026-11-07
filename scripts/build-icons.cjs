// Package the unchanged transparent brand artwork into white browser icons.
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('playwright');
(async()=>{
 const publicDir=path.resolve(__dirname,'../public');
 const executablePath=process.env.CLASSIO_CHROMIUM_PATH||(fs.existsSync('/usr/bin/chromium')?'/usr/bin/chromium':undefined);
 const browser=await chromium.launch({executablePath,headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage();
  const source='data:image/png;base64,'+fs.readFileSync(path.join(publicDir,'ClassioLogo-transparent.png')).toString('base64');
  for(const size of[32,192,512]){
   const data=await page.evaluate(async({source,size})=>{
    const image=new Image();image.src=source;await image.decode();
    const canvas=document.createElement('canvas');canvas.width=canvas.height=size;
    const ctx=canvas.getContext('2d');ctx.fillStyle='#fff';ctx.fillRect(0,0,size,size);
    const padding=Math.max(2,Math.round(size*.07));ctx.drawImage(image,padding,padding,size-padding*2,size-padding*2);
    return canvas.toDataURL('image/png').split(',')[1];
   },{source,size});
   fs.writeFileSync(path.join(publicDir,'favicon-'+size+'.png'),Buffer.from(data,'base64'));
  }
  const png=fs.readFileSync(path.join(publicDir,'favicon-32.png')),head=Buffer.alloc(22);
  head.writeUInt16LE(1,2);head.writeUInt16LE(1,4);head[6]=head[7]=32;head.writeUInt16LE(1,10);head.writeUInt16LE(32,12);head.writeUInt32LE(png.length,14);head.writeUInt32LE(22,18);
  fs.writeFileSync(path.join(publicDir,'favicon.ico'),Buffer.concat([head,png]));
 }finally{await browser.close();}
})().catch(error=>{console.error(error.message);process.exitCode=1;});

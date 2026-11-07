const {validDesign}=require("../src/shared/avatar-schema.js");
function sharedDesign(input) {
  if(!validDesign(input)) return null;
  const match=/^data:image\/(png|jpeg|webp);base64,([a-z0-9+/=]+)$/i.exec(input.texture);if(!match)return null;
  const bytes=Buffer.from(match[2],'base64'),real=match[1]==='png'?bytes.subarray(0,8).toString('hex')==='89504e470d0a1a0a':match[1]==='jpeg'?bytes.subarray(0,3).toString('hex')==='ffd8ff':bytes.subarray(0,4).toString()==='RIFF'&&bytes.subarray(8,12).toString()==='WEBP';if(!real)return null;
  return {kind:input.kind,region:input.region,name:input.name.trim(),baseStyle:input.baseStyle,texture:input.texture,...Object.fromEntries(['length','fringe','textureAmount'].map(key=>[key,Math.min(1,Math.max(0,Number(input[key])||0))]))};
}
module.exports={sharedDesign};

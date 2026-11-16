// Byte signatures are checked independently of the caller-supplied MIME type.
function inspectImage(bytes){
 const at=(offset,text)=>Array.from(text).every((letter,index)=>bytes[offset+index]===letter.charCodeAt(0));
 if(bytes.length>=16&&at(0,'\x89PNG\r\n\x1a\n'))return {mime:'image/png'};
 if(bytes.length>=4&&bytes[0]===255&&bytes[1]===216&&bytes[2]===255)return {mime:'image/jpeg'};
 if(bytes.length>=16&&at(0,'RIFF')&&at(8,'WEBP'))return {mime:'image/webp',animated:at(12,'VP8X')&&!!(bytes[20]&2)};
 if(bytes.length>=14&&(at(0,'GIF87a')||at(0,'GIF89a'))){
   let cursor=13+(bytes[10]&128?3*(2**((bytes[10]&7)+1)):0),frames=0;
   const skip=()=>{while(cursor<bytes.length){const size=bytes[cursor++];if(!size)return;cursor+=size;}};
   while(cursor<bytes.length){const marker=bytes[cursor++];if(marker===0x3b)break;if(marker===0x21){cursor++;skip();}else if(marker===0x2c){if(cursor+9>bytes.length)return null;const flags=bytes[cursor+8];cursor+=9+(flags&128?3*2**((flags&7)+1):0);cursor++;skip();frames++;if(frames>1)return{mime:'image/gif',animated:true};}else return null;}
   return frames?{mime:'image/gif',animated:false}:null;
 }
 return null;
}
module.exports={inspectImage};

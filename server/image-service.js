const {geminiBlocked,noteGeminiFailure}=require("./gemini-pool");
const { providerKey,geminiCredentials,credentialEnvironment } = require("./ai-settings");
class ImageServiceError extends Error { constructor(status,message){super(message);this.status=status;} }
const dataImage = value => typeof value === "string" && value.length <= 4200000 && /^data:image\/(png|jpeg|webp);base64,[a-z0-9+/=]+$/i.test(value);
function imageCapabilities(env=process.env) {
  return { imageEditing: env.GEMINI_ENABLED === "true" && !!providerKey("gemini",env) && /^[\w.-]+$/.test(env.GEMINI_IMAGE_MODEL || ""), backgroundRemoval: !!env.REMOVE_BG_API_KEY?.trim() };
}
async function imageOperation(input,{env=process.env,fetcher=global.fetch}={}) {
  const capability=imageCapabilities(env);
  if (input.action === "remove-background") {
    if (!capability.backgroundRemoval) throw new ImageServiceError(503,"Background removal requires the server’s REMOVE_BG_API_KEY setting. The original image is kept.");
    if (!dataImage(input.image)) throw new ImageServiceError(400,"Choose a supported image under 3 MB.");
    const form=new FormData();form.set("image_file_b64",input.image.split(",")[1]);form.set("size","auto");form.set("format","png");
    const response=await fetcher("https://api.remove.bg/v1.0/removebg",{method:"POST",headers:{"X-Api-Key":env.REMOVE_BG_API_KEY},body:form,signal:AbortSignal.timeout(40000)});
    if (!response.ok) throw new ImageServiceError(response.status===429?429:502,"The background-removal service could not process this image. Try again later; the original is unchanged.");
    const bytes=Buffer.from(await response.arrayBuffer());
    if(bytes.length>3000000 || bytes.subarray(0,8).toString("hex")!=="89504e470d0a1a0a")throw new ImageServiceError(502,"The service returned an unsupported cutout.");
    return {image:"data:image/png;base64,"+bytes.toString("base64")};
  }
  if (!capability.imageEditing) throw new ImageServiceError(503,"Sketch refinement needs an enabled Gemini image-generation model (GEMINI_IMAGE_MODEL) on the server. Drawing, manual adjustments and private drafts remain available.");
  if (!['avatar','generate'].includes(input.action) || typeof input.prompt!=="string" || !input.prompt.trim() || input.prompt.length>4000 || (input.image && !dataImage(input.image))) throw new ImageServiceError(400,"Add design instructions and a supported sketch.");
  const instructions=input.action==='avatar' ? "Refine this user's original sketch into a square, seamless avatar fabric or hair texture. Preserve the sketch's colors, drawn shapes and intended details. Cute rounded game-like original art, clean edges. Output only the flat texture, without a person, body, face or mannequin. Preserve lettering and symbols the user drew or requested; do not invent extra emblems or text. Transfer the drawn decorative details to the flat front material rather than printing the garment outline on the fabric. Keep it appropriate for a general educational community. This texture will be fitted onto an existing 3D character. " : "Create an accurate educational illustration. Preserve requested labels and scientific meaning. ";
  const parts=[{text:instructions+input.prompt}];
  if(input.image){const match=/^data:(image\/\w+);base64,(.+)$/.exec(input.image);parts.push({inlineData:{mimeType:match[1],data:match[2]}});}
  let last;const deadline=Date.now()+45000;
  for(const credential of geminiCredentials(env)){
    const active=credentialEnvironment(env,credential),blocked=geminiBlocked(active);
    if(blocked){last=new ImageServiceError(blocked.code==='PROVIDER_RATE_LIMIT'?429:503,'The image model is cooling down. Your original design is kept.');continue;}
    try{
      const response=await fetcher(`https://generativelanguage.googleapis.com/v1beta/models/${env.GEMINI_IMAGE_MODEL}:generateContent`,{method:'POST',headers:{'Content-Type':'application/json','x-goog-api-key':credential.key},body:JSON.stringify({contents:[{role:'user',parts}],generationConfig:{responseModalities:['TEXT','IMAGE']}}),signal:AbortSignal.timeout(Math.max(1,deadline-Date.now()))});
      if(!response.ok){const seconds=Number(response.headers?.get?.('retry-after'))||60;noteGeminiFailure(active,response.status,seconds);const error=new ImageServiceError(response.status===429?429:502,'The image model could not complete the design. Your sketch is preserved; try again later.');if(![401,403,429].includes(response.status)&&response.status<500)throw error;last=error;continue;}
      const data=await response.json();if(data.promptFeedback?.blockReason||['SAFETY','PROHIBITED_CONTENT','BLOCKLIST'].includes(data.candidates?.[0]?.finishReason))throw new ImageServiceError(422,'This image could not be generated. Adjust the design or description.');
      const image=data.candidates?.[0]?.content?.parts?.map(part=>part.inlineData||part.inline_data).find(part=>/^image\/(png|jpeg|webp)$/.test(part?.mimeType||part?.mime_type));
      if(!image||typeof image.data!=='string'||image.data.length>4000000||!image.data.length||!/^[a-z0-9+/=]+$/i.test(image.data))throw new ImageServiceError(422,'The image model did not produce a supported image. Adjust the sketch or description.');
      return{image:`data:${image.mimeType||image.mime_type};base64,${image.data}`};
    }catch(error){if(error instanceof ImageServiceError)throw error;last=new ImageServiceError(503,'The image service did not respond. Your original sketch is kept.');}
    if(Date.now()>=deadline)break;
  }
  throw last||new ImageServiceError(503,'No image-generation credential is available.');
}
module.exports={imageCapabilities,imageOperation,ImageServiceError,dataImage};

import schema from "./shared/avatar-schema.js";
export const DESIGN_LIMIT=4;
const {validDesign}=schema;
export {validDesign};
export async function fitDesignTexture(src, background="#ffffff") {
  if (!/^data:image\/(?:png|jpeg|webp);base64,[a-z0-9+/=]+$/i.test(src || "") || src.length > 4200000) throw new Error("The design service returned an unsupported image.");
  const image = new Image(); image.src = src; await image.decode();
  if (image.naturalWidth < 64 || image.naturalHeight < 64) throw new Error("This design image is too small to fit the character.");
  const canvas = document.createElement("canvas"); canvas.width = canvas.height = 256;
  const context = canvas.getContext("2d"); context.fillStyle=/^#[a-f0-9]{6}$/i.test(background)?background:"#ffffff";context.fillRect(0,0,256,256);context.drawImage(image,0,0,256,256);
  for(const quality of [.85,.65,.45]){const result=canvas.toDataURL("image/webp",quality);if(result.length<45000)return result;}
  throw new Error("The design is too detailed to save. Simplify it and try again.");
}
export function applyDesign(character, design) {
  if (!validDesign(design)) throw new Error("This design must be converted and validated before it can be worn.");
  if (design.kind === "hair") return {...character,hairStyle:design.baseStyle,customHair:design};
  if (design.region === "hat") return {...character,hat:design.baseStyle,customHat:design};
  if (design.region === "shoes") return {...character,shoeStyle:design.baseStyle,customShoes:design};
  if (design.region === "glasses") return {...character,glasses:design.baseStyle,customGlasses:design};
  if (design.region === "necklace") return {...character,necklace:design.baseStyle,customNecklace:design};
  if (design.region === "earring") return {...character,earring:design.baseStyle,customEarring:design};
  if (design.region === "bottom") return {...character,customBottom:design,pantsStyle:design.baseStyle === 1 ? 1 : 0};
  if (design.region === "accessory") return {...character,customAccessory:design};
  return {...character,topStyle:design.baseStyle,customClothing:design};
}

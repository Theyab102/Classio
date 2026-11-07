const ranges={top:[0,15],bottom:[0,1],hair:[0,16],hat:[1,8],shoes:[0,3],glasses:[1,7],accessory:[0,3],necklace:[1,5],earring:[1,6]};
const kinds={clothing:['top','bottom'],hair:['hair'],hat:['hat'],shoes:['shoes'],glasses:['glasses'],accessory:['accessory','necklace','earring']};
function validDesign(value){
  if(!value||!kinds[value.kind]?.includes(value.region)||typeof value.name!=='string'||!value.name.trim()||value.name.length>60||!/^data:image\/(?:png|webp|jpeg);base64,[a-z0-9+/=]+$/i.test(value.texture||'')||value.texture.length>=45000)return false;
  const range=ranges[value.region];if(!Number.isInteger(value.baseStyle)||value.baseStyle<range[0]||value.baseStyle>range[1])return false;
  return ['length','fringe','textureAmount'].every(key=>value[key]===undefined||(typeof value[key]==='number'&&Number.isFinite(value[key])&&value[key]>=0&&value[key]<=1));
}
module.exports={validDesign};

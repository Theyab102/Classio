export const DEFAULT_CHARACTER = { skin: "#edc5a1", hair: "#493326", hairStyle: 1, eyes: "#487b9b", top: "#528ca6", bg: "#e6f2fb", mouth: 0, eyebrow: 0, eyeShape: 0, topStyle: 1, blush: true, lips: false, freckles: false, lipColor: "#b96565", hat: 0, hatColor: "#739db8", glasses: 0, glassesColor: "#394957", facialHair: 0, necklace: 0, necklaceColor: "#c99d48", earring: 0, earringColor: "#c99d48", name: "", pantsStyle:0, shoeStyle:0, accessoryStyle:0, bottom:"#3b4b60", shoes:"#e3d4b7", outfitAccent:"#fffaf1", accessoryColor:"#b66957" };
export const AVATAR_OPTIONS = {
  hairStyle: ["Close crop", "Side part", "Fringe", "Textured", "Long", "Wavy", "Ponytail", "Bun", "Afro", "Braids", "Curly", "Locs", "Faux hawk", "Two buns", "Mohawk", "Dreadlocks", "Swept back", "Chin-length bob", "Pixie cut", "Curtain bangs", "Shag", "Long bob", "Low ponytail", "Pigtails", "High puff", "Cornrows", "Side braid", "Half-up bun", "Wolf cut", "Buzz fade", "Braided buns", "Round bob", "Bald"],
  eyeShape: ["Round", "Soft", "Lashes", "Relaxed", "Joyful", "Bright", "Almond", "Sleepy"], mouth: ["Smile", "Thoughtful", "Surprised", "Smirk", "Neutral", "Beaming", "Delighted", "Toothy grin", "Small smile", "Pout"],
  eyebrow: ["Arched", "Straight", "Full", "Soft angle", "Curious", "Gentle", "Rounded", "Bold arch", "Tapered"],
  topStyle: ["T-shirt", "Hoodie", "Jacket", "Tank top", "Suit", "Knot top", "Polo", "Jersey", "Striped top", "Dress", "Blazer", "Graphic tee", "Varsity", "Wrap top", "Turtleneck", "Denim", "Cable-knit sweater", "Oxford shirt", "Camp shirt", "Cardigan", "Dungarees", "Raglan tee", "Raincoat", "Kimono cardigan", "Sailor shirt", "Puff-sleeve blouse", "Peplum top", "Fleece vest", "Patterned sweater", "Chore jacket", "Pinafore", "Sweater vest", "Track jacket", "Bowling shirt", "Ruffle blouse", "Longline coat"],
  pantsStyle: ["Match outfit", "Shorts", "Jeans", "Joggers", "Pleated skirt", "Cargo pants", "Overalls", "Wide-leg trousers", "Chinos"],
  shoeStyle: ["Sneakers", "Ankle boots", "Ballet flats", "Sandals", "High-tops", "Loafers", "Rain boots", "Slippers", "Moccasins", "Lace-up shoes"],
  accessoryStyle: ["None", "Backpack", "Satchel", "Little wings", "Curled tail", "Watch", "Wristbands", "Star badge", "Crossbody bag"],
  hat: ["None", "Beanie", "Cap", "Cowboy", "Graduation", "Crown", "Party hat", "Headband", "Bucket hat", "Beret", "Newsboy cap", "Sun hat", "Knit cap", "Ribbon bow", "Visor", "Earflap hat", "Flower crown", "Top hat"],
  glasses: ["None", "Round", "Rectangle", "Cat-eye", "Sunglasses", "Hearts", "Aviator", "Monocle", "Browline", "Oversized square", "Sport frames", "Slim oval", "Star frames"],
  facialHair: ["None", "Stubble", "Moustache", "Short beard", "Full beard", "Goatee", "Sideburns", "Chin patch"], necklace: ["None", "Chain", "Pendant", "Beads", "Choker", "Pearl", "Scarf", "Bow tie", "Ribbon", "Crescent"],
  earring: ["None", "Studs", "Drops", "Hoops", "Stars", "Dangles", "Crosses", "Crescents", "Flowers", "Double hoops"],
};
export const AVATAR_COLORS = {
  skin: ["#f4d6c8", "#fddbb4", "#e8a87c", "#c68642", "#a0693a", "#8d5524", "#6b3a1f", "#49291c"],
  hair: ["#242424", "#493326", "#815533", "#b8783e", "#d6b774", "#eee0bb", "#ad543e", "#777f86"],
  top: ["#528ca6", "#264658", "#738e72", "#b66957", "#b99d61", "#ededdf", "#787e8b", "#a46378"],
  bottom: ["#3b4b60", "#264658", "#706c61", "#314f45", "#9b7860", "#e1d7c5", "#573f56", "#28313a"],
  shoes: ["#e3d4b7", "#fffaf1", "#3b4b60", "#805b42", "#b66957", "#738e72", "#a46378", "#282e32"],
  outfitAccent: ["#fffaf1", "#264658", "#738e72", "#b66957", "#d6b774", "#a46378"],
  accessoryColor: ["#b66957", "#528ca6", "#738e72", "#b99d61", "#a46378", "#264658"],
  eyes: ["#373b38", "#775b3b", "#4f7766", "#487b9b"], bg: ["#e6f2fb", "#f4ead8", "#e3eddc", "#f4e1dc", "#e7e9ed", "#273e4b"],
};
export function randomCharacter(current, random = Math.random, preset = "any") {
  const choose = list => list[Math.min(list.length - 1, Math.floor(random() * list.length))];
  const next = { ...current };
  for (const [key, list] of Object.entries(AVATAR_COLORS)) next[key] = choose(list);
  for (const key of ["hairStyle", "eyeShape", "mouth", "eyebrow", "topStyle", "pantsStyle", "shoeStyle"]) next[key] = Math.floor(random() * AVATAR_OPTIONS[key].length);
  const presets = { male: {hairStyle:[0,1,2,3,8,10,12,14,16,18,19,20,25,28,29,32],topStyle:[0,1,2,4,6,7,10,11,12,14,15,16,17,18,19,21,22,27,28,29,31,32,33,35]}, female:{hairStyle:[2,4,5,6,7,9,10,11,13,15,17,18,19,20,21,22,23,24,26,27,28,30,31,32],topStyle:[0,1,2,3,5,9,10,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34,35]} };
  if(presets[preset]) for(const key of ["hairStyle","topStyle"]) next[key] = choose(presets[preset][key]);
  for(const key of ["customClothing","customHair","customBottom","customShoes","customHat","customGlasses","customNecklace","customEarring","customAccessory"]) next[key] = null;
  next.hat = random() < .75 ? 0 : 1 + Math.floor(random() * (AVATAR_OPTIONS.hat.length-1));
  next.glasses = random() < .6 ? 0 : 1 + Math.floor(random() * (AVATAR_OPTIONS.glasses.length-1));
  next.accessoryStyle = random()<.6?0:choose(AVATAR_OPTIONS.accessoryStyle.map((_,i)=>i).slice(1));
  next.necklace = random()<.65?0:choose(AVATAR_OPTIONS.necklace.map((_,i)=>i).slice(1));
  next.earring = random()<.65?0:choose(AVATAR_OPTIONS.earring.map((_,i)=>i).slice(1));
  next.blush = random() > .3; next.freckles = random() > .65;
  return next;
}

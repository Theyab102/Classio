export const DEFAULT_CHARACTER = { skin: "#edc5a1", hair: "#493326", hairStyle: 1, eyes: "#487b9b", top: "#528ca6", bg: "#e6f2fb", mouth: 0, eyebrow: 0, eyeShape: 0, topStyle: 1, blush: true, lips: false, freckles: false, lipColor: "#b96565", hat: 0, hatColor: "#739db8", glasses: 0, glassesColor: "#394957", facialHair: 0, necklace: 0, necklaceColor: "#c99d48", earring: 0, earringColor: "#c99d48", name: "" };
export const AVATAR_OPTIONS = {
  hairStyle: ["Close crop", "Side part", "Fringe", "Textured", "Long", "Wavy", "Ponytail", "Bun", "Afro", "Braids", "Curly", "Locs", "Faux hawk", "Two buns", "Mohawk", "Dreadlocks", "Swept back"],
  eyeShape: ["Round", "Soft", "Lashes", "Relaxed", "Joyful", "Bright"], mouth: ["Smile", "Thoughtful", "Surprised", "Smirk", "Neutral", "Beaming", "Delighted"],
  eyebrow: ["Arched", "Straight", "Full", "Soft angle", "Curious", "Gentle"],
  topStyle: ["T-shirt", "Hoodie", "Jacket", "Tank top", "Suit", "Knot top", "Polo", "Jersey", "Striped top", "Dress", "Blazer", "Graphic tee", "Varsity", "Wrap top", "Turtleneck", "Denim"],
  hat: ["None", "Beanie", "Cap", "Cowboy", "Graduation", "Crown", "Party hat", "Headband", "Bucket hat"],
  glasses: ["None", "Round", "Rectangle", "Cat-eye", "Sunglasses", "Hearts", "Aviator", "Monocle"],
  facialHair: ["None", "Stubble", "Moustache", "Short beard", "Full beard", "Goatee"], necklace: ["None", "Chain", "Pendant", "Beads", "Choker", "Pearl"],
  earring: ["None", "Studs", "Drops", "Hoops", "Stars", "Dangles", "Crosses"],
};
export const AVATAR_COLORS = {
  skin: ["#f4d6c8", "#fddbb4", "#e8a87c", "#c68642", "#a0693a", "#8d5524", "#6b3a1f", "#49291c"],
  hair: ["#242424", "#493326", "#815533", "#b8783e", "#d6b774", "#eee0bb", "#ad543e", "#777f86"],
  top: ["#528ca6", "#264658", "#738e72", "#b66957", "#b99d61", "#ededdf", "#787e8b", "#a46378"],
  eyes: ["#373b38", "#775b3b", "#4f7766", "#487b9b"], bg: ["#e6f2fb", "#f4ead8", "#e3eddc", "#f4e1dc", "#e7e9ed", "#273e4b"],
};
export function randomCharacter(current, random = Math.random, preset = "any") {
  const choose = list => list[Math.min(list.length - 1, Math.floor(random() * list.length))];
  const next = { ...current };
  for (const [key, list] of Object.entries(AVATAR_COLORS)) next[key] = choose(list);
  for (const key of ["hairStyle", "eyeShape", "mouth", "eyebrow", "topStyle"]) next[key] = Math.floor(random() * AVATAR_OPTIONS[key].length);
  const presets = { male: {hairStyle:[0,1,2,3,8,10,12,14,16],topStyle:[0,1,2,4,6,7,10,11,12,14,15]}, female:{hairStyle:[2,4,5,6,7,9,10,11,13,15],topStyle:[0,1,2,3,5,9,10,13,14,15]} };
  if(presets[preset]) for(const key of ["hairStyle","topStyle"]) next[key] = choose(presets[preset][key]);
  for(const key of ["customClothing","customHair","customBottom","customShoes","customHat","customGlasses","customNecklace","customEarring","customAccessory"]) next[key] = null;
  next.hat = random() < .75 ? 0 : 1 + Math.floor(random() * 8);
  next.glasses = random() < .6 ? 0 : 1 + Math.floor(random() * 7);
  next.blush = random() > .3; next.freckles = random() > .65;
  return next;
}

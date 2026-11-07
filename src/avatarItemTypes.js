export const ITEM_TYPES = {
  clothing:{label:'Clothing',fits:[['Shirt / T-shirt','top',0],['Hoodie','top',1],['Jacket','top',2],['Pants','bottom',0],['Shorts','bottom',1],['Dress','top',9]]},
  hair:{label:'Hairstyle',fits:['Close crop','Side part','Fringe','Textured','Long','Wavy','Ponytail','Bun','Afro','Braids','Curly','Locs','Faux hawk','Two buns','Mohawk','Dreadlocks','Swept back'].map((name,i)=>[name,'hair',i])},
  hat:{label:'Hat',fits:[['Beanie','hat',1],['Cap','hat',2],['Cowboy hat','hat',3],['Graduation cap','hat',4],['Crown','hat',5],['Party hat','hat',6],['Headband','hat',7],['Bucket hat','hat',8]]},
  shoes:{label:'Shoes',fits:[['Sneakers','shoes',0],['Boots','shoes',1],['Loafers','shoes',2],['Rounded flats','shoes',3]]},
  glasses:{label:'Glasses',fits:[['Round','glasses',1],['Rectangle','glasses',2],['Cat-eye','glasses',3],['Sunglasses','glasses',4],['Hearts','glasses',5],['Aviator','glasses',6],['Monocle','glasses',7]]},
  accessory:{label:'Accessory',fits:[['Shirt badge','accessory',0],['Backpack','accessory',1],['Wings','accessory',2],['Tail','accessory',3],['Chain','necklace',1],['Pendant','necklace',2],['Beads','necklace',3],['Stud earrings','earring',1],['Drop earrings','earring',2],['Hoop earrings','earring',3]]},
};

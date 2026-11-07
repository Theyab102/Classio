import {applyDesign,validDesign} from './avatarDesigns';
import {randomCharacter} from './avatarOptions';
const design={kind:'hat',region:'hat',name:'Blue hat',baseStyle:2,texture:'data:image/png;base64,YWJj'};
test('wearable conversion selects the actual body slot and preserves existing custom items',()=>{
  const hat=applyDesign({name:'Sam',customShoes:{name:'My shoes'}},design);
  expect(hat.hat).toBe(2);expect(hat.customHat).toEqual(design);expect(hat.customShoes.name).toBe('My shoes');
  const glasses=applyDesign(hat,{...design,kind:'glasses',region:'glasses',baseStyle:1});expect(glasses.customHat).toEqual(design);expect(glasses.glasses).toBe(1);
});
test('invalid shape conversions and outside image URLs cannot become wearable or shared items',()=>{
  expect(validDesign({...design,region:'shoes'})).toBe(false);expect(validDesign({...design,baseStyle:9})).toBe(false);expect(validDesign({...design,texture:'https://example.test/image.png'})).toBe(false);expect(()=>applyDesign({}, {...design,baseStyle:30})).toThrow();
});
test('male and female preset randomizers differ while keeping the saved design collection',()=>{
  const current={name:'Sam',designs:[design],customHat:design};const male=randomCharacter(current,()=>.5,'male'),female=randomCharacter(current,()=>.5,'female');expect(male.hairStyle).not.toBe(female.hairStyle);expect(male.designs).toEqual([design]);expect(male.name).toBe('Sam');expect(male.customHat).toBeNull();
});

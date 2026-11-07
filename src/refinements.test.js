import {normalizeStyle,contrastRatio} from './styleConfig';
import {deckGeometry} from './deckGeometry';
import {parseAvatarFit} from './avatarFit';
test('generated styles reject executable values, bound tokens, and repair both theme contrasts',()=>{
 const style=normalizeStyle({font:'url(bad)',radius:900,spacing:-3,paper:'script',light:{surface:'#ffffff',text:'#ffffff'},dark:{surface:'#ffffff',text:'#ffffff'}});
 expect(style.font).toBe('sans');expect(style.radius).toBe(22);expect(style.spacing).toBe(.85);expect(style.paper).toBe('plain');
 for(const mode of ['light','dark'])for(const key of ['text','muted','accent'])expect(contrastRatio(style[mode][key],style[mode].surface)).toBeGreaterThanOrEqual(4.5);
});
test('every theme keeps title, body and inline image in separate export rectangles',()=>{
 const overlaps=(a,b)=>a.x<b.x+b.w&&a.x+a.w>b.x&&a.y<b.y+b.h&&a.y+a.h>b.y;
 for(const layout of ['lecture','editorial','visual'])for(const type of ['title','content'])for(const imagePlacement of ['left','right','below']){
  const boxes=deckGeometry({type,image:'fixture',imagePlacement},layout);expect(overlaps(boxes.title,boxes.image)).toBe(false);expect(overlaps(boxes.body,boxes.image)).toBe(false);expect(overlaps(boxes.title,boxes.body)).toBe(false);for(const box of [boxes.title,boxes.body,boxes.image]){expect(box.x+box.w).toBeLessThanOrEqual(13.334);expect(box.y+box.h).toBeLessThanOrEqual(7.5);}
 }
});
test('sketch analysis accepts confident compatible fits and rejects guesses outside the catalog',()=>{
 expect(parseAvatarFit('{"fitIndex":0,"confidence":0.9,"length":2}', 'clothing').length).toBe(1);
 expect(parseAvatarFit('{"fitIndex":0,"confidence":0.3}', 'hat')).toBeNull();
 expect(parseAvatarFit('{"fitIndex":999,"confidence":1}', 'shoes')).toBeNull();
});

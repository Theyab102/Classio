import {parseDeckResponse} from './DeckDesign';
import {deckGeometry} from './deckGeometry';
import {createSlide, updateSlideById, imageCrop} from './presentationModel';
import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import SlideEditor from './SlideEditor';
import {requestImage} from './imageService';
jest.mock('./imageService',()=>({requestImage:jest.fn()}));
beforeEach(()=>{global.fetch=jest.fn(async()=>({ok:true,json:async()=>({imageGeneration:false})}));jest.clearAllMocks();});
jest.mock('./pdf',()=>({openPdf:jest.fn(),loadPdfLibrary:jest.fn()}));
jest.mock('./SlideCanvasEditor',()=>()=>null);

test('new slide types survive AI parsing and stable IDs isolate late edits from reordering',()=>{
  for(const type of ['image','timeline','summary','comparison']){
    const slide=parseDeckResponse(JSON.stringify({slides:[{type,title:'Motion',bullets:['First','Next']}]}),1).slides[0];
    expect(slide.type).toBe(type==='comparison'?'two-col':type);
  }
  const a=createSlide('content'),b=createSlide('image');
  expect(updateSlideById([b,a],a.id,{title:'Changed'}).map(s=>s.title)).toEqual([b.title,'Changed']);
  expect(updateSlideById([b],a.id,{title:'Late'})).toEqual([b]);
});
test('a failed apply retains the complete slide draft and reports the failure',async()=>{
 const slide=createSlide('content'),onClose=jest.fn();
 render(<SlideEditor slide={slide} index={0} total={1} theme={{}} layout="lecture" onApply={()=>Promise.reject(new Error('unavailable'))} onClose={onClose}/>);
 fireEvent.change(screen.getByLabelText('Slide title'),{target:{value:'My unsaved explanation'}});
 fireEvent.click(screen.getByRole('button',{name:'Apply changes'}));
 await waitFor(()=>expect(screen.getByRole('alert')).toHaveTextContent('Your draft is still here'));
 expect(screen.getByLabelText('Slide title')).toHaveValue('My unsaved explanation');expect(onClose).not.toHaveBeenCalled();
});
test('custom image geometry stays in the slide and crop uses a shared bounded rectangle',()=>{
  const slide={type:'image',image:'data:image/png;base64,YWJj',imagePlacement:'full',geometry:{image:{x:99,y:-1,w:99,h:99}}};
  const box=deckGeometry(slide).image;
  expect(box.x).toBeGreaterThanOrEqual(0);expect(box.y).toBeGreaterThanOrEqual(0);
  expect(box.x+box.w).toBeLessThanOrEqual(13.333);expect(box.y+box.h).toBeLessThanOrEqual(7.5);
  expect(imageCrop({x:-1,y:4,w:9,h:9})).toEqual({x:0,y:.95,w:1,h:.05});
  expect(imageCrop(null)).toEqual({x:0,y:0,w:1,h:1});
  expect(deckGeometry({type:'quote',quote:'A thought'}).body).toMatchObject({y:2.2,h:2.7});
});
test('illustrations use the secure dedicated image route without enabling sketch editing',async()=>{
 global.fetch=jest.fn(async()=>({ok:true,json:async()=>({imageGeneration:true,imageEditing:false})}));
 requestImage.mockResolvedValue('data:image/png;base64,YWJj');const apply=jest.fn();
 render(<SlideEditor slide={createSlide('content')} index={0} total={1} theme={{}} layout="lecture" onApply={apply} onClose={jest.fn()}/>);
 fireEvent.click(screen.getByRole('button',{name:'AI tools',exact:true}));
 await waitFor(()=>expect(screen.getByRole('button',{name:'Generate illustration'})).toBeEnabled());
 fireEvent.click(screen.getByRole('button',{name:'Generate illustration',exact:true}));
 await waitFor(()=>expect(requestImage).toHaveBeenCalledWith(expect.objectContaining({action:'generate',prompt:expect.any(String)}),expect.any(AbortSignal)));
 await waitFor(()=>expect(screen.getByRole('button',{name:'Apply changes'})).toBeEnabled());fireEvent.click(screen.getByRole('button',{name:'Apply changes'}));
 await waitFor(()=>expect(apply).toHaveBeenCalledWith(expect.objectContaining({image:'data:image/png;base64,YWJj'})));
});
test('unconfigured image generation is disabled while image uploads and search stay available',async()=>{
 render(<SlideEditor slide={createSlide('content')} index={0} total={1} theme={{}} layout="lecture" onApply={jest.fn()} onClose={jest.fn()}/>);
 fireEvent.click(screen.getByRole('button',{name:'AI tools',exact:true}));
 await screen.findByText(/Image generation is currently unavailable/);
 expect(screen.getByRole('button',{name:'Generate illustration'})).toBeDisabled();
 fireEvent.click(screen.getByRole('button',{name:'Images & media'}));
 expect(screen.getByRole('button',{name:'Upload image'})).toBeEnabled();
 expect(screen.getByRole('button',{name:'Search images'})).toBeEnabled();
 expect(requestImage).not.toHaveBeenCalled();
});

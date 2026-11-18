import {useRef} from 'react';
import {fireEvent,render,screen} from '@testing-library/react';
import useDocumentGestures from './useDocumentGestures';
function Fixture({onPage,scope=1,fingerInk=false}){const ref=useRef();useDocumentGestures(ref,onPage,scope);return <div ref={ref} aria-label="Viewport" style={{overflow:'auto'}}><canvas aria-label="Surface" className="studio-ink" data-finger-drawing={fingerInk}/><button>Tool</button></div>;}
const point=(x,y=100)=>({clientX:x,clientY:y});
function touch(kind,points,target=screen.getByLabelText('Surface')){fireEvent(target,new TouchEvent(kind,{bubbles:true,cancelable:true,touches:points}));}
function swipe(count=1,dx=-90,dy=0){const points=Array.from({length:count},(_,i)=>point(180+i*30,100));touch('touchstart',points);touch('touchmove',points.map(p=>point(p.clientX+dx,p.clientY+dy)));touch('touchend',[]);}
test.each([1,2])('%i fingers turn exactly one page and ignore lifting the remaining finger',count=>{
 const onPage=jest.fn();render(<Fixture onPage={onPage}/>);swipe(count);touch('touchend',[]);expect(onPage).toHaveBeenCalledTimes(1);expect(onPage).toHaveBeenCalledWith(1);
 swipe(count,90);expect(onPage).toHaveBeenLastCalledWith(-1);
});
test('vertical scrolling and cancelled swipes do not turn a page',()=>{
 const onPage=jest.fn();render(<Fixture onPage={onPage}/>);const view=screen.getByLabelText('Viewport');
 swipe(2,0,-90);expect(view.scrollTop).toBe(90);expect(onPage).not.toHaveBeenCalled();
 touch('touchstart',[point(180)]);touch('touchmove',[point(80)]);touch('touchcancel',[]);touch('touchend',[]);expect(onPage).not.toHaveBeenCalled();
});
test('a pinch is never interpreted as a page turn',()=>{
 const onPage=jest.fn();render(<Fixture onPage={onPage}/>);touch('touchstart',[point(140),point(180)]);touch('touchmove',[point(10),point(100)]);touch('touchend',[]);expect(onPage).not.toHaveBeenCalled();
});
test('zoomed material pans before flipping and a new swipe at its edge can flip',()=>{
 const onPage=jest.fn();render(<Fixture onPage={onPage}/>);const view=screen.getByLabelText('Viewport');let left=0;
 Object.defineProperties(view,{clientWidth:{value:100},scrollWidth:{value:400},scrollLeft:{get:()=>left,set:value=>{left=Math.max(0,Math.min(300,value));}}});
 swipe();expect(left).toBe(90);expect(onPage).not.toHaveBeenCalled();left=300;swipe();expect(onPage).toHaveBeenCalledWith(1);
});
test('explicit finger ink keeps a single finger, but two fingers still navigate',()=>{
 const onPage=jest.fn();render(<Fixture onPage={onPage} fingerInk/>);swipe(1);expect(onPage).not.toHaveBeenCalled();swipe(2);expect(onPage).toHaveBeenCalledWith(1);
});
test('tool controls and gestures from an earlier page cannot navigate',()=>{
 const onPage=jest.fn(),{rerender}=render(<Fixture onPage={onPage}/>);touch('touchstart',[point(180)],screen.getByRole('button'));touch('touchmove',[point(80)],screen.getByRole('button'));touch('touchend',[],screen.getByRole('button'));expect(onPage).not.toHaveBeenCalled();
 touch('touchstart',[point(180)]);touch('touchmove',[point(80)]);rerender(<Fixture onPage={onPage} scope={2}/>);touch('touchend',[]);expect(onPage).not.toHaveBeenCalled();
});

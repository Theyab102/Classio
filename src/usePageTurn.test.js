import {useRef,useState} from 'react';
import {act,fireEvent,render,screen} from '@testing-library/react';
import usePageTurn from './usePageTurn';
jest.mock('./pageTurnPreviews',()=>({copyPageSnapshot:node=>node.cloneNode(true)}));
let animations,reduce,observer,viewportWidth;
beforeEach(()=>{
 animations=[];reduce=false;viewportWidth=600;
 window.matchMedia=()=>({matches:reduce,addEventListener(){},removeEventListener(){}});
 global.ResizeObserver=class{constructor(callback){observer=callback;}observe(){}disconnect(){}};
 jest.spyOn(HTMLElement.prototype,'getBoundingClientRect').mockReturnValue({left:0,top:0,width:600,height:400});
 Object.defineProperties(HTMLElement.prototype,{clientWidth:{configurable:true,get:()=>viewportWidth},clientHeight:{configurable:true,get:()=>400}});
 HTMLElement.prototype.animate=jest.fn((frames,settings)=>{let finish;const finished=new Promise(resolve=>finish=resolve),animation={finished,cancel:jest.fn(),frames,settings,finish};animations.push(animation);return animation;});
});
afterEach(()=>{jest.restoreAllMocks();delete HTMLElement.prototype.animate;delete HTMLElement.prototype.clientWidth;delete HTMLElement.prototype.clientHeight;delete window.matchMedia;delete global.ResizeObserver;});
function Fixture({onChange=()=>{},ready=true,failed=false,resetKey='file:fit'}){
 const viewport=useRef(),surface=useRef(),[page,setPage]=useState(2);
 const turn=usePageTurn({viewport,surface,page,total:3,ready,failed,resetKey,onChange:next=>{onChange(next);setPage(next);},preview:async next=>{const node=document.createElement('div');node.textContent='Preview '+next;return{node,width:600,height:400};}});
 return <><button onClick={()=>turn.gesture.drag(-90)}>Drag left</button><button onClick={()=>turn.gesture.drag(90)}>Drag right</button><button onClick={turn.gesture.cancel}>Cancel</button><button onClick={()=>turn.goTo(page+1)}>Next</button><button onClick={()=>turn.goTo(page-1)}>Previous</button><div ref={viewport} aria-label="Viewport"><div ref={surface} data-page-turn-current>Page {page}</div></div></>;
}
async function finish(){await act(async()=>{for(const animation of animations)animation.finish();await Promise.resolve();});}
test('drag tracks the finger and the adjacent page; release sends both planes in the same direction',async()=>{
 render(<Fixture/>);fireEvent.click(screen.getByText('Drag left'));await act(async()=>{});
 expect(document.querySelector('.page-turn-outgoing').style.transform).toBe('translate3d(-90px,0,0)');expect(document.querySelector('.page-turn-incoming').style.transform).toBe('translate3d(510px,0,0)');expect(screen.getByText('Preview 3')).toBeInTheDocument();
 fireEvent.click(screen.getByText('Next'));expect(animations[0].frames.at(-1).transform).toBe('translate3d(-600px,0,0)');expect(animations[1].frames.at(-1).transform).toBe('translate3d(0px,0,0)');
 await finish();expect(document.querySelector('.page-turn-overlay')).toBeNull();expect(screen.getByText('Page 3')).toBeInTheDocument();
});
test('the previous page enters from the left while the current page moves right',async()=>{
 render(<Fixture/>);fireEvent.click(screen.getByText('Drag right'));await act(async()=>{});expect(document.querySelector('.page-turn-incoming').style.transform).toBe('translate3d(-510px,0,0)');fireEvent.click(screen.getByText('Previous'));
 expect(animations[0].frames.at(-1).transform).toBe('translate3d(600px,0,0)');await finish();
});
test('cancelled and boundary swipes spring back without changing pages',async()=>{
 const changed=jest.fn();render(<Fixture onChange={changed}/>);fireEvent.click(screen.getByText('Drag left'));fireEvent.click(screen.getByText('Cancel'));expect(animations[0].frames.at(-1).transform).toBe('translate3d(0px,0,0)');await finish();expect(changed).not.toHaveBeenCalled();
 fireEvent.click(screen.getByText('Next'));await finish();changed.mockClear();fireEvent.click(screen.getByText('Drag left'));expect(document.querySelector('.page-turn-outgoing').style.transform).toBe('translate3d(-18px,0,0)');fireEvent.click(screen.getByText('Next'));await finish();expect(changed).not.toHaveBeenCalled();
});
test('a slow page stays covered by the outgoing snapshot until its real content is ready',async()=>{
 const {rerender}=render(<Fixture/>);fireEvent.click(screen.getByText('Drag left'));rerender(<Fixture ready={false}/>);fireEvent.click(screen.getByText('Next'));expect(screen.getByLabelText('Viewport')).toHaveAttribute('data-page-turn','waiting');expect(animations).toHaveLength(0);
 rerender(<Fixture ready/>);expect(animations).toHaveLength(2);await finish();
});
test('late completion of an interrupted turn cannot remove the newer turn',async()=>{
 render(<Fixture/>);fireEvent.click(screen.getByText('Next'));const old=[...animations];fireEvent.click(screen.getByText('Previous'));expect(old[0].cancel).toHaveBeenCalled();await act(async()=>{old.forEach(animation=>animation.finish());await Promise.resolve();});expect(document.querySelector('.page-turn-overlay')).not.toBeNull();await finish();
});
test('reduced motion changes pages without moving copies',()=>{
 reduce=true;render(<Fixture/>);fireEvent.click(screen.getByText('Drag left'));fireEvent.click(screen.getByText('Next'));expect(screen.getByText('Page 3')).toBeInTheDocument();expect(document.querySelector('.page-turn-overlay')).toBeNull();expect(animations).toHaveLength(0);
});
test('zoom/file changes and unmount remove every temporary copy',()=>{
 const{rerender,unmount}=render(<Fixture/>);fireEvent.click(screen.getByText('Drag left'));rerender(<Fixture resetKey="file:200"/>);expect(document.querySelector('.page-turn-overlay')).toBeNull();fireEvent.click(screen.getByText('Drag right'));unmount();expect(document.querySelector('.page-turn-overlay')).toBeNull();
});

test('resizing the viewport cancels an obsolete gesture without changing the page',()=>{
 const changed=jest.fn();render(<Fixture onChange={changed}/>);fireEvent.click(screen.getByText('Drag left'));viewportWidth=500;act(()=>observer());expect(document.querySelector('.page-turn-overlay')).toBeNull();expect(changed).not.toHaveBeenCalled();
});

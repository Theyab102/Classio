import {fireEvent,render,screen} from '@testing-library/react';
import {InkLayer,InkToolbar,useInkTool} from './DocumentInk';
import {useState} from 'react';
test('Clear ink can be undone as a whole operation and new drawing invalidates redo',()=>{
 function Fixture(){const ink=useInkTool(),[strokes,set]=useState([{tool:'pen',points:[[0,0]]},{tool:'pen',points:[[1,1]]}]);return <><InkToolbar ink={ink} strokes={strokes} onChange={set}/><output aria-label="Stroke count">{strokes.length}</output></>;}
 render(<Fixture/>);fireEvent.click(screen.getByText('Clear ink'));expect(screen.getByLabelText('Stroke count')).toHaveTextContent('0');
 fireEvent.click(screen.getByRole('button',{name:'Undo ink'}));expect(screen.getByLabelText('Stroke count')).toHaveTextContent('2');
 fireEvent.click(screen.getByRole('button',{name:'Redo ink'}));expect(screen.getByLabelText('Stroke count')).toHaveTextContent('0');
});
test('Undo and Redo stay visible with Drawing options collapsed',()=>{
 function Fixture(){const ink=useInkTool();return <InkToolbar ink={ink} strokes={[]} onChange={()=>{}}/>;}
 render(<Fixture/>);expect(screen.getByRole('button',{name:'Undo ink'})).toBeVisible();expect(screen.getByRole('button',{name:'Redo ink'})).toBeVisible();
 expect(screen.getByRole('button',{name:'Drawing options'})).toHaveAttribute('aria-expanded','false');
});

describe('stylus and finger input ownership',()=>{
 let originalObserver,callbacks,paint;
 beforeEach(()=>{
  localStorage.removeItem('classio_tablet_mode_v1');callbacks=[];paint=jest.fn();originalObserver=global.ResizeObserver;global.ResizeObserver=class{constructor(callback){callbacks.push(callback);}observe(){}disconnect(){}};
  jest.spyOn(HTMLCanvasElement.prototype,'getBoundingClientRect').mockReturnValue({left:0,top:0,width:700,height:400});
  jest.spyOn(HTMLCanvasElement.prototype,'getContext').mockReturnValue({save(){},restore(){},clearRect:jest.fn(),beginPath(){},moveTo(){},lineTo(){},stroke:paint});
  HTMLCanvasElement.prototype.setPointerCapture=jest.fn();
 });
 afterEach(()=>{global.ResizeObserver=originalObserver;jest.restoreAllMocks();delete HTMLCanvasElement.prototype.setPointerCapture;});
 function Fixture(){const ink=useInkTool('one'),[strokes,set]=useState([]);return <><InkToolbar ink={ink} strokes={strokes} onChange={set}/><InkLayer ink={ink} strokes={strokes} onChange={set}/><output aria-label="Saved strokes">{JSON.stringify(strokes)}</output></>;}
 function pointer(kind,id,type='pen',x=20){const event=new MouseEvent(kind,{bubbles:true,button:0,clientX:x,clientY:20});Object.defineProperties(event,{pointerId:{value:id},pointerType:{value:type}});fireEvent(screen.getByLabelText('Ink layer'),event);}
 test('only a finished stylus stroke is saved; cancellation discards its preview',()=>{
  render(<Fixture/>);fireEvent.click(screen.getByRole('button',{name:'Pen',exact:true}));
  pointer('pointerdown',1);pointer('pointermove',1,'pen',70);pointer('pointercancel',1);pointer('pointerup',1);
  expect(screen.getByLabelText('Saved strokes')).toHaveTextContent('[]');
  pointer('pointerdown',2);pointer('pointermove',2,'pen',70);pointer('pointerup',2);
  expect(JSON.parse(screen.getByLabelText('Saved strokes').textContent)).toHaveLength(1);
 });
 test('fingers navigate by default; a second finger discards provisional finger ink',()=>{
  render(<Fixture/>);fireEvent.click(screen.getByRole('button',{name:'Pen',exact:true}));
  pointer('pointerdown',1,'touch');pointer('pointermove',1,'touch',80);pointer('pointerup',1,'touch');
  expect(screen.getByLabelText('Saved strokes')).toHaveTextContent('[]');
  fireEvent.click(screen.getByLabelText('Draw with finger'));
  pointer('pointerdown',2,'touch');pointer('pointermove',2,'touch',50);pointer('pointerdown',3,'touch');pointer('pointerup',2,'touch');pointer('pointerup',3,'touch');
  expect(screen.getByLabelText('Saved strokes')).toHaveTextContent('[]');
  pointer('pointerdown',4,'touch');pointer('pointermove',4,'touch',80);pointer('pointerup',4,'touch');
  expect(JSON.parse(screen.getByLabelText('Saved strokes').textContent)).toHaveLength(1);
 });
 test('mouse-emulated tablet strokes write with Pen; hover and Scroll do not write',()=>{
  render(<Fixture/>);pointer('pointerdown',1,'mouse');pointer('pointerup',1,'mouse');expect(screen.getByLabelText('Saved strokes')).toHaveTextContent('[]');
  fireEvent.click(screen.getByRole('button',{name:'Pen',exact:true}));pointer('pointermove',2,'mouse',80);expect(screen.getByLabelText('Saved strokes')).toHaveTextContent('[]');
  pointer('pointerdown',2,'mouse');pointer('pointermove',2,'mouse',80);pointer('pointerup',2,'mouse',90);const saved=JSON.parse(screen.getByLabelText('Saved strokes').textContent);expect(saved).toHaveLength(1);expect(saved[0].points).toHaveLength(3);
 });
 test('drawing tablet mode handles touch-emulated pen input without enabling finger drawing',()=>{
  render(<Fixture/>);fireEvent.click(screen.getByRole('button',{name:'Pen',exact:true}));fireEvent.click(screen.getByLabelText('Drawing tablet mode'));expect(screen.getByLabelText('Draw with finger')).not.toBeChecked();
  pointer('pointerdown',1,'touch');pointer('pointermove',1,'touch',80);pointer('pointerup',1,'touch',80);expect(JSON.parse(screen.getByLabelText('Saved strokes').textContent)).toHaveLength(1);
 });
 test('a queued resize redraw cannot resurrect a stroke after Undo',()=>{
  render(<Fixture/>);fireEvent.click(screen.getByRole('button',{name:'Pen',exact:true}));pointer('pointerdown',1);pointer('pointermove',1,'pen',70);pointer('pointerup',1);
  const queued=callbacks.at(-1);fireEvent.click(screen.getByRole('button',{name:'Undo ink'}));expect(screen.getByLabelText('Saved strokes')).toHaveTextContent('[]');
  paint.mockClear();queued();expect(paint).not.toHaveBeenCalled();
 });
 test('Undo clears the previous bitmap even while the page has zero layout bounds',()=>{
  render(<Fixture/>);fireEvent.click(screen.getByRole('button',{name:'Pen',exact:true}));pointer('pointerdown',1);pointer('pointermove',1,'pen',70);pointer('pointerup',1);
  const canvas=screen.getByLabelText('Ink layer'),context=canvas.getContext('2d');context.clearRect.mockClear();paint.mockClear();
  HTMLCanvasElement.prototype.getBoundingClientRect.mockReturnValue({width:0,height:0});fireEvent.click(screen.getByRole('button',{name:'Undo ink'}));
  expect(context.clearRect).toHaveBeenCalledWith(0,0,canvas.width,canvas.height);expect(paint).not.toHaveBeenCalled();
 });
 test('pen input waits until the moving page has settled',()=>{
  render(<Fixture/>);fireEvent.click(screen.getByRole('button',{name:'Pen',exact:true}));const view=screen.getByLabelText('Ink layer').parentElement;view.dataset.pageTurn='turning';
  pointer('pointerdown',1);pointer('pointermove',1,'pen',70);pointer('pointerup',1);expect(screen.getByLabelText('Saved strokes')).toHaveTextContent('[]');
  delete view.dataset.pageTurn;pointer('pointerdown',2);pointer('pointermove',2,'pen',70);pointer('pointerup',2);expect(JSON.parse(screen.getByLabelText('Saved strokes').textContent)).toHaveLength(1);
 });
});

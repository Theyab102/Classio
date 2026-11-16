import {fireEvent,render,screen} from '@testing-library/react';
import {InkToolbar,useInkTool} from './DocumentInk';
import {useState} from 'react';
test('Clear ink can be undone as a whole operation and new drawing invalidates redo',()=>{
 function Fixture(){const ink=useInkTool(),[strokes,set]=useState([{tool:'pen',points:[[0,0]]},{tool:'pen',points:[[1,1]]}]);return <><InkToolbar ink={ink} strokes={strokes} onChange={set}/><output aria-label="Stroke count">{strokes.length}</output></>;}
 render(<Fixture/>);fireEvent.click(screen.getByText('Clear ink'));expect(screen.getByLabelText('Stroke count')).toHaveTextContent('0');
 fireEvent.click(screen.getByText('Undo ink'));expect(screen.getByLabelText('Stroke count')).toHaveTextContent('2');
 fireEvent.click(screen.getByText('Redo ink'));expect(screen.getByLabelText('Stroke count')).toHaveTextContent('0');
});

import {fireEvent,render,screen} from '@testing-library/react';
import useSourcePageKeys from './useSourcePageKeys';
function Fixture({onPage,enabled=true}){useSourcePageKeys(onPage,enabled);return <><nav className="classio-navigation"><a href="#source">Source</a></nav><input aria-label="Search source"/><textarea aria-label="Question"/><div contentEditable suppressContentEditableWarning aria-label="Editor"/><canvas aria-label="Ink" tabIndex={0}/></>;}
test('arrows flip source pages even when Source link or annotation canvas has focus',()=>{
 const onPage=jest.fn();render(<Fixture onPage={onPage}/>);fireEvent.keyDown(screen.getByRole('link'),{key:'ArrowRight'});fireEvent.keyDown(screen.getByLabelText('Ink'),{key:'ArrowLeft'});expect(onPage.mock.calls).toEqual([[1],[-1]]);
});
test('editing, modifiers, dialogs and disabled viewers do not change pages',()=>{
 const onPage=jest.fn(),view=render(<Fixture onPage={onPage}/>);
 for(const label of ['Search source','Question','Editor'])fireEvent.keyDown(screen.getByLabelText(label),{key:'ArrowRight'});
 fireEvent.keyDown(window,{key:'ArrowLeft',altKey:true});fireEvent.keyDown(window,{key:'ArrowLeft',shiftKey:true});expect(onPage).not.toHaveBeenCalled();
 const dialog=document.createElement('div');dialog.setAttribute('role','dialog');dialog.setAttribute('aria-modal','true');document.body.append(dialog);fireEvent.keyDown(window,{key:'ArrowRight'});expect(onPage).not.toHaveBeenCalled();dialog.remove();
 view.rerender(<Fixture enabled={false} onPage={onPage}/>);fireEvent.keyDown(window,{key:'ArrowRight'});expect(onPage).not.toHaveBeenCalled();
});

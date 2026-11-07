import {historyReducer} from './useActionHistory';
test('separate editor histories support undo/redo and new edits discard only their own future',()=>{
  let a={past:[],present:'A',future:[]},b={past:[],present:'Other editor',future:[]};a=historyReducer(a,{value:'B'});a=historyReducer(a,{type:'undo'});expect(a.present).toBe('A');expect(b.present).toBe('Other editor');a=historyReducer(a,{type:'redo'});expect(a.present).toBe('B');a=historyReducer(historyReducer(a,{type:'undo'}),{value:'C'});expect(a.future).toEqual([]);
});

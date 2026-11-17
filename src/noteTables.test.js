import {Editor} from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import {TableKit} from '@tiptap/extension-table';
import {fittedColumns,NoteTable,insertNoteTable,mergeNextNoteCell,selectNoteTable} from './noteTables';

beforeAll(()=>{global.ResizeObserver=class {observe(){}disconnect(){}};});
test('column sizes remain within the editor and retain usable minimum widths',()=>{
  const values=fittedColumns([10,5000,10],300);
  expect(values.reduce((sum,value)=>sum+value,0)).toBeCloseTo(300);
  expect(Math.min(...values)).toBeGreaterThanOrEqual(72);
  expect(fittedColumns([0,0,0],600)).toEqual([200,200,200]);
});
test.each(['Backspace','Delete'])('selecting an entire table and pressing %s removes only that table and supports Undo',key=>{
 const editor=new Editor({extensions:[StarterKit,TableKit.configure({table:false}),NoteTable],content:'<p>Keep before.</p><table><tr><td><p>Delete table content</p></td><td><p>Second cell</p></td></tr></table><p>Keep after.</p>'});
 try{let pos;editor.state.doc.descendants((node,offset)=>{if(node.isText&&node.text==='Delete table content')pos=offset;});editor.commands.setTextSelection(pos);expect(selectNoteTable(editor)).toBe(true);expect(editor.state.selection.node.type.name).toBe('table');editor.commands.keyboardShortcut(key);expect(editor.getHTML()).not.toContain('<table');expect(editor.getText()).toContain('Keep before.');expect(editor.getText()).toContain('Keep after.');editor.commands.undo();expect(editor.getHTML()).toContain('Delete table content');}finally{editor.destroy();}
});
test('touch-friendly adjacent merging keeps both cell contents and can be split again',()=>{
 const editor=new Editor({extensions:[StarterKit,TableKit.configure({table:false}),NoteTable],content:'<table><tr><td><p>First</p></td><td><p>Second</p></td></tr></table>'});
 try{editor.commands.setTextSelection(4);expect(mergeNextNoteCell(editor)).toBe(true);expect(editor.getJSON().content[0].content[0].content).toHaveLength(1);expect(editor.getText()).toContain('First');expect(editor.getText()).toContain('Second');editor.commands.splitCell();expect(editor.getJSON().content[0].content[0].content).toHaveLength(2);}finally{editor.destroy();}
});
test('inserting inside a heading keeps its entire word together and repeated insertion is outside the table',()=>{
  const editor=new Editor({extensions:[StarterKit,TableKit.configure({table:false}),NoteTable],content:'<h1>Projectile Motion</h1>'});
  try {
    editor.commands.setTextSelection(7);insertNoteTable(editor);
    expect(editor.getJSON().content[0].content[0].text).toBe('Projectile Motion');
    expect(editor.getJSON().content[1].type).toBe('table');expect(editor.getHTML()).not.toContain('<th');
    editor.commands.setTextSelection(23); // Inside the first cell of the table.
    insertNoteTable(editor);
    const root=document.createElement('div');root.innerHTML=editor.getHTML();
    expect(root.querySelectorAll('table')).toHaveLength(2);expect(root.querySelector('table table')).toBeNull();
    editor.commands.undo();expect(editor.getJSON().content.filter(item=>item.type==='table')).toHaveLength(1);
  } finally {editor.destroy();}
});

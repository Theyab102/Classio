import {Table,TableView,createTable} from '@tiptap/extension-table';
import {Plugin} from '@tiptap/pm/state';
import {TableMap,cellAround,selectionCell} from '@tiptap/pm/tables';
import {closeHistory} from '@tiptap/pm/history';

export function fittedColumns(weights, width, minimum=72) {
  const min=Math.min(minimum,width/weights.length),remaining=Math.max(0,width-min*weights.length);
  const extras=weights.map(value=>Math.max(0,(Number(value)||width/weights.length)-min));
  const total=extras.reduce((sum,value)=>sum+value,0);
  return extras.map(value=>min+remaining*(total?value/total:1/weights.length));
}
function weightsFor(node) {
  const weights=[];
  node.firstChild?.forEach(cell=>{for(let i=0;i<cell.attrs.colspan;i++)weights.push(cell.attrs.colwidth?.[i]||100);});
  return weights;
}
class FittedTableView extends TableView {
  constructor(...args) {
    super(...args);
    this.fit();
    this.observer=new ResizeObserver(()=>{
      // Row height changes are an expected consequence of wrapping. Only a
      // changed container width needs a fit; defer writes out of delivery.
      if(this.dom.clientWidth===this.lastWidth)return;
      cancelAnimationFrame(this.frame);this.frame=requestAnimationFrame(()=>this.fit());
    });
    this.observer.observe(this.dom);
  }
  fit() {
    this.lastWidth=this.dom.clientWidth;
    const width=this.lastWidth||600,values=fittedColumns(weightsFor(this.node),width);
    const stacked=width<values.length*72;
    this.dom.dataset.stacked=String(stacked);this.table.setAttribute('aria-label',stacked?'Wide table, rows stacked to fit this screen':'Notes table');
    this.table.style.width='100%';this.table.style.minWidth='0';
    [...this.colgroup.children].forEach((col,i)=>{col.style.minWidth='0';col.style.width=`${values[i]/width*100}%`;});
  }
  update(node) {const updated=super.update(node);if(updated)this.fit();return updated;}
  destroy() {this.observer.disconnect();cancelAnimationFrame(this.frame);}
}

// Resize adjacent columns together. The stock plugin grows the whole table and
// permits 25px cells; these bounded drags keep the table inside its editor.
function resizePlugin() {
  let cancelDrag=()=>{};
  const boundary=(view,event)=>{
    if(!view.editable)return null;
    const cell=event.target.closest?.('td,th');
    if(!cell||!view.dom.contains(cell)||cell.closest('.tableWrapper')?.dataset.stacked==='true'||Math.abs(event.clientX-cell.getBoundingClientRect().right)>8)return null;
    const $cell=cellAround(view.state.doc.resolve(view.posAtDOM(cell,0)));
    if(!$cell)return null;
    const table=$cell.node(-1),map=TableMap.get(table),col=map.colCount($cell.pos-$cell.start(-1))+$cell.nodeAfter.attrs.colspan-1;
    return col<map.width-1?{cell,table,map,col,start:$cell.start(-1)}:null;
  };
  return new Plugin({
    props:{handleDOMEvents:{
      pointermove(view,event) {view.dom.classList.toggle('resize-cursor',!!boundary(view,event));return false;},
      pointerleave(view) {view.dom.classList.remove('resize-cursor');return false;},
      pointerdown(view,event) {
        const found=boundary(view,event);if(!found||event.button!==0)return false;
        cancelDrag();event.preventDefault();
        const {cell,table,map,col,start}=found,dom=cell.closest('table'),width=dom.getBoundingClientRect().width;
        const weights=fittedColumns(weightsFor(table),width),pair=weights[col]+weights[col+1],minimum=Math.min(72,width/map.width),origin=event.clientX;
        const doc=view.state.doc,win=view.dom.ownerDocument.defaultView;
        let next=weights.slice();
        const preview=()=>[...dom.querySelectorAll(':scope > colgroup > col')].forEach((node,i)=>node.style.width=`${next[i]/width*100}%`);
        const move=event=>{next=weights.slice();next[col]=Math.max(minimum,Math.min(pair-minimum,weights[col]+event.clientX-origin));next[col+1]=pair-next[col];preview();};
        const cleanup=()=>{win.removeEventListener('pointermove',move);win.removeEventListener('pointerup',finish);win.removeEventListener('pointercancel',cancel);view.dom.classList.remove('resize-cursor');cancelDrag=()=>{};};
        const cancel=()=>{next=weights;preview();cleanup();};
        const finish=()=>{
          cleanup();
          // Do not apply positions captured before a concurrent document edit.
          if(view.isDestroyed||view.state.doc!==doc)return;
          const tr=closeHistory(view.state.tr),seen=new Set();
          for(const pos of map.map) {
            if(seen.has(pos))continue;seen.add(pos);
            const node=table.nodeAt(pos),first=map.colCount(pos);
            tr.setNodeMarkup(start+pos,null,{...node.attrs,colwidth:next.slice(first,first+node.attrs.colspan).map(value=>Math.round(value))});
          }
          view.dispatch(tr);
        };
        cancelDrag=cancel;win.addEventListener('pointermove',move);win.addEventListener('pointerup',finish);win.addEventListener('pointercancel',cancel);
        return true;
      },
    }},
    view:()=>({destroy:()=>cancelDrag()}),
  });
}
export const NoteTable=Table.extend({
  addProseMirrorPlugins(){return [...this.parent(),resizePlugin()];},
}).configure({resizable:false,renderWrapper:true,HTMLAttributes:{role:"table"},View:FittedTableView});

export function insertNoteTable(editor,{rows=3,cols=3,header=false,html}={}) {
  const {$to}=editor.state.selection;
  let depth=1;
  for(let i=$to.depth;i>0;i--)if($to.node(i).type.name==='table'){depth=i;break;}
  // Insert after the containing table or complete top-level text block, never
  // between letters of a heading or inside the current cell.
  const pos=$to.depth?$to.after(depth):$to.pos;
  const content=html||createTable(editor.schema,rows,cols,header).toJSON();
  return editor.chain().focus().command(({tr})=>{closeHistory(tr);return true;}).insertContentAt(pos,content).run();
}

export function adjacentNoteCells(editor) {
  try {
    const cell=selectionCell(editor.state),table=cell.node(-1),start=cell.start(-1),map=TableMap.get(table),rect=map.findCell(cell.pos-start);
    if(rect.right>=map.width)return null;
    const next=map.map[rect.top*map.width+rect.right],other=map.findCell(next);
    // With row spans, merge only a genuine adjacent pair, not a larger rectangle.
    if(other.top!==rect.top||other.bottom!==rect.bottom)return null;
    return {anchorCell:cell.pos,headCell:start+next};
  } catch {return null;}
}
export function mergeNextNoteCell(editor) {
  const pair=adjacentNoteCells(editor);
  return pair?editor.chain().focus().setCellSelection(pair).mergeCells().run():false;
}

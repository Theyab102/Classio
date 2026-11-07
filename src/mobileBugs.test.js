import {act,render,screen} from '@testing-library/react';
import {useRef} from 'react';
import {designFromInstructions,normalizeDeckDesign,deckStyleVariables} from './deckStyle';
import {decorateDeckExport} from './deckExportDesign';
import {chooseSourceFigure} from './sourceFigures';
import {enrichNoteImages} from './noteFigures';
import useVisualViewport from './useVisualViewport';
import {contrastRatio} from './styleConfig';
jest.mock('./pdf',()=>({openPdf:jest.fn(),loadPdfLibrary:jest.fn()}));
test('style instructions produce a real supported pattern with bounded safe values',()=>{
 const design=designFromInstructions({name:'Class notes',bg:'#ffffff',text:'#ffffff',font:'url(bad)',radius:200},'Use a blue and white checkerboard with handwritten headings');
 expect(design.pattern).toBe('checkerboard');expect(design.font).toBe('handwritten');expect(design.radius).toBe(20);expect(design.text).not.toBe('#ffffff');expect(deckStyleVariables(design,design).backgroundImage).toContain('conic-gradient');
 expect(()=>normalizeDeckDesign(null)).toThrow();
 expect(designFromInstructions({pattern:'checkerboard'},'without a checkerboard').pattern).toBe('none');
 const patterned=normalizeDeckDesign({bg:'#ffffff',text:'#757575',sub:'#757575',accent:'#ffffff',pattern:'checkerboard',patternColor:'#000000',patternOpacity:.16});
 for(const ink of [patterned.text,patterned.sub,patterned.accent]){expect(contrastRatio(ink,'#ffffff')).toBeGreaterThanOrEqual(4.5);expect(contrastRatio(ink,'#d6d6d6')).toBeGreaterThanOrEqual(4.5);}
});
test('checkerboard exports are actual bounded native shapes rather than a renamed color palette',()=>{
 const shapes=[],pptx={ShapeType:{rect:'rect',line:'line',ellipse:'ellipse'}},slide={addShape:(type,options)=>shapes.push({type,...options}),addText:jest.fn()};
 const design=normalizeDeckDesign({name:'Check',pattern:'checkerboard'});decorateDeckExport(pptx,slide,{layout:'editorial',index:0,total:1,cover:true,bg:'FFFFFF',accent:'286A9B',sub:'253648',design});
 expect(shapes.length).toBe(195);expect(shapes.every(shape=>shape.w>0&&shape.h>0&&shape.x+shape.w<=13.334&&shape.y+shape.h<=7.501)).toBe(true);
});
test('source selection honors validated source IDs and falls back to a relevant actual file figure',()=>{
 const figures=[{src:'data:image/png;base64,YWJj',title:'Projectile trajectory',context:'projectile motion parabolic path'},{src:'data:image/png;base64,ZGVm',title:'Terminal velocity',context:'falling drag forces'}];
 expect(chooseSourceFigure({sourceFigure:1},figures)).toBe(figures[1]);expect(chooseSourceFigure({sourceFigure:999,title:'Projectile trajectory'},figures)).toBe(figures[0]);
});
test('actual file figures appear beside explanations even when the model omitted markers',async()=>{
 const image={src:'data:image/png;base64,YWJj',title:'Projectile trajectory',context:'Projectile motion gravity parabola',credit:'Source slide 3'},search=jest.fn();
 const result=await enrichNoteImages('# Physics\n\n## Projectile motion\n\nA projectile follows a curved trajectory under gravity.',{fileObj:{name:'Physics.pptx'},sourceFigures:[image],sourceOnly:true,useWeb:true,search});
 const root=document.createElement('div');root.innerHTML=result.html;expect(root.querySelector('img').getAttribute('src')).toBe(image.src);expect(root.querySelector('h2').nextElementSibling.nextElementSibling.tagName).toBe('IMG');expect(search).not.toHaveBeenCalled();
});
test('source figure IDs survive math recovery and place the actual image by its subject',async()=>{
 const image={src:'data:image/png;base64,YWJj',title:'Forces in motion',context:'force motion',credit:'Source slide 1'};
 const result=await enrichNoteImages('# Forces and motion\n\nA force changes motion.\n\n## Units\n\nThe force unit is the newton.\n\n## Suggested Visual Searches\n\n[[SOURCE_FIGURE:0]]',{fileObj:{name:'Forces.pptx'},sourceFigures:[image],sourceOnly:true,useWeb:true});
 const root=document.createElement('div');root.innerHTML=result.html;expect(root.querySelector('h1').nextElementSibling.nextElementSibling.getAttribute('src')).toBe(image.src);expect(result.html).not.toContain('SOURCE_FIGURE');expect(result.html).not.toContain('Suggested Visual Searches');expect(root.querySelector('[data-classio-equation]')).toBeNull();
});
test('keyboard-sized visual viewports update modal bounds and restore on dismissal',()=>{
 const previous=Object.getOwnPropertyDescriptor(window,'visualViewport'),view=new EventTarget();view.height=window.innerHeight;view.offsetTop=0;Object.defineProperty(window,'visualViewport',{configurable:true,value:view});
 function Dialog(){const ref=useRef(null);useVisualViewport(ref);return<div ref={ref} data-testid="viewport"><textarea aria-label="Draft"/></div>;}
 try{render(<Dialog/>);screen.getByLabelText('Draft').focus();act(()=>{view.height=300;view.offsetTop=45;view.dispatchEvent(new Event('resize'));});const dialog=screen.getByTestId('viewport');expect(dialog.style.getPropertyValue('--visual-height')).toBe('300px');expect(dialog.style.getPropertyValue('--visual-top')).toBe('45px');expect(dialog.dataset.keyboardOpen).toBe('true');act(()=>{view.height=window.innerHeight;view.offsetTop=0;view.dispatchEvent(new Event('resize'));});expect(dialog.dataset.keyboardOpen).toBe('false');}finally{if(previous)Object.defineProperty(window,'visualViewport',previous);else delete window.visualViewport;}
});

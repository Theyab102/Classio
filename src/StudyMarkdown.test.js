jest.mock('./pdf',()=>({openPdf:jest.fn(),loadPdfLibrary:jest.fn()}));
import {render,screen} from '@testing-library/react';
import StudyMarkdown from './StudyMarkdown';
test('a graph survives ordinary Markdown sanitation and renders actual axes and a curve',()=>{
 const {container}=render(<StudyMarkdown text={'## End behavior\n\nThe parabola rises at both ends.\n\n```classio-graph\n'+JSON.stringify({expression:'x^2-3*x+2',title:'Quadratic graph',xMin:-2,xMax:5})+'\n```'}/>);
 expect(screen.getByRole('img',{name:'Quadratic graph'})).toBeInTheDocument();expect(container.querySelector('svg path[stroke="#286a9b"]').getAttribute('d')).toContain('L');expect(screen.getByText('The parabola rises at both ends.')).toBeInTheDocument();expect(container.textContent).not.toContain('"expression"');
});
test('unsafe equations become an actionable message rather than executable markup',()=>{
 const {container}=render(<StudyMarkdown text={'```classio-graph\n'+JSON.stringify({expression:'window.alert(1)',title:'<img src=x onerror=alert(1)>'})+'\n```'}/>);expect(screen.getByText(/This graph could not be drawn/)).toBeInTheDocument();expect(container.querySelector('img,script,svg')).toBeNull();
});

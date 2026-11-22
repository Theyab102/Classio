import {render,screen} from '@testing-library/react';
import {WorkspaceLink} from './WorkspaceNavigation';
import StudyMarkdown from './StudyMarkdown';
jest.mock('./pdf',()=>({openPdf:jest.fn(),loadPdfLibrary:jest.fn()}));
test('material previews keep readable formatting without nesting links inside their file link',()=>{
 render(<WorkspaceLink to={{screen:'file',fileId:'one',fileTab:'notes'}} aria-label="Open notes"><StudyMarkdown inline allowLinks={false} text="Read [the guide](https://example.test) or https://example.test. **Keep this emphasis.**"/></WorkspaceLink>);
 const link=screen.getByRole('link',{name:'Open notes'});expect(link.querySelectorAll('a')).toHaveLength(0);expect(link.querySelector('strong')).toHaveTextContent('Keep this emphasis.');expect(link.textContent).toContain('the guide');
});
test('links remain available in normal study content',()=>{
 render(<StudyMarkdown text="Read [the guide](https://example.test)."/>);expect(screen.getByRole('link',{name:'the guide'})).toHaveAttribute('href','https://example.test');
});

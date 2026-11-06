import {render,screen} from '@testing-library/react';
import StudyMarkdown,{studyMarkdownHtml} from './StudyMarkdown';

test('assistant messages render real Markdown tables, code, and emphasis',()=>{
  render(<StudyMarkdown text={'## Energy\n\n**Conserved**\n\n| Form | Example |\n| --- | --- |\n| Kinetic | Motion |\n\n```js\nconst energy = 1;\n```'}/>);
  expect(screen.getByRole('heading',{name:'Energy'})).toBeInTheDocument();
  expect(screen.getByRole('cell',{name:'Motion'})).toBeInTheDocument();
  expect(screen.getByText('Conserved').tagName).toBe('STRONG');
  expect(screen.getByText('const energy = 1;')).toBeInTheDocument();
});
test('math is rendered while raw HTML, unsafe links, and remote images cannot execute or load',()=>{
  const html=studyMarkdownHtml('$E=mc^2$\n\n\\(x^2\\)\n\n<img src="https://tracker.invalid/pixel" onerror="alert(1)">\n\n![tracker](https://tracker.invalid/image)\n\n[unsafe](javascript:alert(1))');
  expect(html).toContain('katex');expect(html).toContain('math');
  expect(html).not.toMatch(/<img|<script|href="javascript:/i);
});
test('mermaid blocks retain the existing interactive renderer',()=>{
  const Diagram=({code})=><p>{code}</p>;
  render(<StudyMarkdown text={'Before\n\n```mermaid\ngraph TD; A-->B;\n```\n\nAfter'} Diagram={Diagram}/>);
  expect(screen.getByText('graph TD; A-->B;')).toBeInTheDocument();
  expect(screen.getByText('After')).toBeInTheDocument();
});

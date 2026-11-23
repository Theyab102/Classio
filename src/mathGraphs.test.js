import {graphFunction,sampleGraph,graphSvg,includeSourceGraph,validateGraphBlocks} from './mathGraphs';
test('graphs evaluate real arithmetic and standard school functions with correct precedence',()=>{
 expect(graphFunction('x^2-3*x+2')(4)).toBe(6);expect(graphFunction('-x^2')(3)).toBe(-9);expect(graphFunction('2^3^2')(0)).toBe(512);expect(graphFunction('sqrt(abs(x))')(-9)).toBe(3);expect(graphFunction('sin(pi/2)')(0)).toBeCloseTo(1);
});
test('JavaScript, property access, arbitrary functions, oversized equations and invalid ranges cannot run',()=>{
 for(const value of ['alert(1)','window.location','constructor(x)','x;fetch(1)','x='.repeat(200),'__proto__(x)'])expect(()=>graphFunction(value)).toThrow();
 expect(()=>sampleGraph({expression:'x',xMin:4,xMax:3})).toThrow();expect(()=>sampleGraph({expression:'sqrt(-1)'})).toThrow();
});
test('the renderer uses real sampled points, labels axes and escapes title markup',()=>{
 const data=sampleGraph({expression:'x^3',xMin:-3,xMax:3});expect(data.points[0]).toEqual([-3,-27]);expect(data.points.at(-1)).toEqual([3,27]);
 const svg=graphSvg({expression:'x^2',title:'<script>no</script>'});expect(svg).toContain('&lt;script&gt;');expect(svg).not.toContain('<script>');expect(svg).toContain('role="img"');expect(svg).toContain('>x</text>');
});
test('a missing graph can be drawn from an explicit source equation without inventing another function',()=>{
 const notes=includeSourceGraph('A complete explanation of end behavior.','Sketch the graph\nf(x) = x² - 3x + 2');expect(notes).toContain('classio-graph');expect(notes.replace(/\s+/g,'')).toContain('x^2-3*x+2');expect(()=>validateGraphBlocks(notes)).not.toThrow();
 expect(includeSourceGraph('Keep explanation.','Sketch a graph from the unreadable diagram.')).toBe('Keep explanation.');
});

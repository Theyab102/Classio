import {inspectImage} from './shared/image-format';
test('image signatures cannot be spoofed by a data URI MIME label',()=>{
 expect(inspectImage(new Uint8Array([1,2,3,4]))).toBeNull();
 const header=new Uint8Array(16);header.set([137,80,78,71,13,10,26,10]);expect(inspectImage(header).mime).toBe('image/png');
});
test('GIF parser distinguishes a static frame from an animated image',()=>{
 const header=[71,73,70,56,57,97,1,0,1,0,0,0,0],frame=[44,0,0,0,0,1,0,1,0,0,2,2,68,1,0];
 expect(inspectImage(new Uint8Array([...header,...frame,59])).animated).toBe(false);
 expect(inspectImage(new Uint8Array([...header,...frame,...frame,59])).animated).toBe(true);
});

import {validateArchive} from './documentArchive';
import JSZip from 'jszip';
test('archive limits reject oversized expanded parts before reading their contents',()=>{
 const read=jest.fn(),file={_data:{uncompressedSize:21*1024*1024},async:read};
 expect(()=>validateArchive({files:{'huge.xml':file}})).toThrow(/size limit/);expect(read).not.toHaveBeenCalled();
 expect(Object.keys(validateArchive({files:{'slide.xml':{_data:{uncompressedSize:200}}}}).files)).toContain('slide.xml');
});
test('actual streamed content is bounded even when archive metadata understates its size',async()=>{
 const zip=new JSZip();zip.file('notes.xml','x'.repeat(100));
 validateArchive(zip,{maxPartBytes:64,maxTotalBytes:128});
 await expect(zip.file('notes.xml').async('string')).rejects.toThrow(/size limit/);
});

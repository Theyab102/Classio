import {linkedFilesFor,setFileLinks,readLinkedStudySource} from './linkedFiles';
import {materialExcerpt} from './assistantRequest';
const pack={id:'pack',name:'Grade 5 revision pack.txt'},key={id:'key',name:'Grade 5 answer key.txt'},other={id:'other',name:'Unrelated private file.txt'};
test('linking and unlinking update both files across folders without altering notes',()=>{
 const folders=[{id:'one',files:[{...pack,notes:'Keep my writing'}]},{id:'two',files:[key,other]}];
 const linked=setFileLinks(folders,'pack',['key','pack','missing','key']);
 expect(linked[0].files[0]).toMatchObject({notes:'Keep my writing',linkedFileIds:['key']});expect(linked[1].files[0].linkedFileIds).toEqual(['pack']);expect(linked[1].files[1]).toBe(other);
 const unlinked=setFileLinks(linked,'key',[]);expect(unlinked[0].files[0].linkedFileIds).toEqual([]);expect(unlinked[1].files[0].linkedFileIds).toEqual([]);expect(folders[0].files[0].linkedFileIds).toBeUndefined();
});
test('old one-way links are visible from either file; links are direct, not transitive',()=>{
 const files=[{...pack,linkedFileIds:['key','deleted']},{...key,linkedFileIds:['other']},other];
 expect(linkedFilesFor(files[1],files).map(file=>file.id)).toEqual(['pack','other']);expect(linkedFilesFor(files[0],files).map(file=>file.id)).toEqual(['key']);
});
test('both sources and their filenames survive repeated downstream context limits',async()=>{
 const files=[{...pack,linkedFileIds:['key']},key,other],read=jest.fn(async file=>file.id==='pack'?'Revision problem: What is 6 times 7?\n\n'.repeat(6000):'Answer key: 42. Explanation: six groups of seven.\n\n'.repeat(6000));
 const text=await readLinkedStudySource(files[0],files,read,120000);
 const bounded=materialExcerpt(text,'six groups',6500);
 expect(bounded.length).toBeLessThanOrEqual(6500);expect(bounded).toContain(pack.name);expect(bounded).toContain(key.name);expect(bounded).toContain('Revision problem:');expect(bounded).toContain('Answer key: 42');expect(read.mock.calls.map(([file])=>file.id)).toEqual(['pack','key']);
});
test('a missing linked original is disclosed and saved notes are labelled as secondary',async()=>{
 const files=[{...pack,linkedFileIds:['key']},key];
 const text=await readLinkedStudySource(files[0],files,async file=>file.id==='pack'?'Read the actual questions.':null);
 expect(text).toContain('Original text unavailable');expect(text).toContain('No readable text is available');expect(text).not.toContain('Answer key: 42');
 const fallback=await readLinkedStudySource({...pack},[{...pack,notes:'Saved explanation.'}],async()=>null);expect(fallback).toContain('Saved notes/transcript');
});
test('a file ID cannot read sources from another workspace',async()=>{
 const read=jest.fn();await expect(readLinkedStudySource(pack,[other],read)).rejects.toThrow('no longer available');expect(read).not.toHaveBeenCalled();
});

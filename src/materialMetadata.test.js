import {inferSubject,materialMatches,materialType} from './materialMetadata';
test('subject inference prioritizes manual choices, then filenames, then content without renaming files',()=>{
  const file={name:'Projectile_Motion AP Physics B Study Notes.pptx',notes:'Biology photosynthesis'};expect(inferSubject(file).label).toBe('Physics');expect(inferSubject({...file,manualSubject:'Science seminar'}).label).toBe('Science seminar');expect(file.name).toContain('Projectile');expect(inferSubject({name:'Reading.pdf',notes:'Mitochondria and cellular respiration'}).label).toBe('Biology');expect(inferSubject({name:'Untitled.pdf'}).label).toBe('General');
});
test('search includes academic subject, MIME type and available extracted content',()=>{
  const file={name:'week-4.pdf',type:'application/pdf',searchText:'A projectile follows a parabolic trajectory.'};expect(materialMatches(file,{name:'School'},'physics pdf')).toBe(true);expect(materialMatches(file,{},'parabolic')).toBe(true);expect(materialMatches(file,{},'biology')).toBe(false);
});
test('link type is based on actual domains and never labels every link YouTube',()=>{
  expect(materialType({youtubeUrl:'https://en.wikipedia.org/wiki/Force',type:'text/youtube'}).label).toBe('Website');expect(materialType({youtubeUrl:'https://youtu.be/abcdefghijk'}).icon).toBe('youtube');expect(materialType({url:'https://youtube.com.attacker.example/x'}).label).toBe('Website');
});

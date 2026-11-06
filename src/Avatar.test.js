import { fireEvent, render, screen, within } from '@testing-library/react';
import { useState } from 'react';
import Avatar from './Avatar';
import AvatarEditor from './AvatarEditor';
import { setLanguages } from './locale';

beforeEach(()=>{setLanguages({interfaceLanguage:'en',replyLanguage:'auto'},{remote:true});});

test('multiple avatars use unique gradient identifiers and reject invalid colour values',()=>{
  const {container}=render(<><Avatar character={{skin:'bad-css-value'}} label="First"/><Avatar label="Second"/></>);
  // Internal SVG paint-server IDs must be unique; these are not accessible UI nodes.
  // eslint-disable-next-line testing-library/no-container, testing-library/no-node-access
  const ids=[...container.querySelectorAll('[id]')].map(item=>item.id);
  expect(new Set(ids).size).toBe(ids.length);
  expect(container.innerHTML).not.toContain('bad-css-value');
  expect(screen.getByRole('img',{name:'First'})).toBeInTheDocument();
});

test('live customization preserves saved legacy attributes and nickname input focus',()=>{
  let value;
  function Harness(){const [character,setCharacter]=useState({name:'Learner',hairStyle:12,unknownLegacySetting:'kept'});value=character;return <AvatarEditor character={character} onChange={setCharacter} onClose={()=>{}}/>;}
  render(<Harness/>);
  const name=screen.getByLabelText('Nickname');name.focus();fireEvent.change(name,{target:{value:'Classio learner'}});
  expect(name).toHaveFocus();
  fireEvent.click(screen.getByRole('button',{name:'Oval',exact:true}));
  expect(value).toMatchObject({faceShape:1,hairStyle:12,name:'Classio learner',unknownLegacySetting:'kept'});
  fireEvent.click(screen.getByRole('tab',{name:'Hair',exact:true}));
  expect(within(screen.getByRole('tabpanel')).getAllByRole('button',{pressed:false}).length).toBeGreaterThan(16);
  fireEvent.click(screen.getByRole('button',{name:'Bun',exact:true}));
  expect(value.hairStyle).toBe(7);
  fireEvent.click(screen.getByRole('tab',{name:'Accessories',exact:true}));
  fireEvent.click(screen.getByRole('button',{name:'Aviator',exact:true}));
  expect(value.glasses).toBe(6);
  expect(value.unknownLegacySetting).toBe('kept');
});

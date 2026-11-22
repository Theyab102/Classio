import {fireEvent,render,screen} from '@testing-library/react';
import {useState} from 'react';
import CountInput from './CountInput';
function Harness(){const [value,setValue]=useState('5');return <form onSubmit={event=>{event.preventDefault();throw new Error('Stepper submitted the form');}}><CountInput label="Number of questions" value={value} onChange={setValue}/></form>;}
test('count is directly editable and stepper buttons do not submit the form',()=>{
  render(<Harness/>);const input=screen.getByRole('spinbutton',{name:'Number of questions'});
  expect(input).toHaveAccessibleDescription('Type a number or use − / + · 1–50');
  fireEvent.change(input,{target:{value:''}});expect(input).toHaveDisplayValue('');
  fireEvent.change(input,{target:{value:'27'}});fireEvent.click(screen.getByRole('button',{name:'Increase number of questions'}));expect(input).toHaveValue(28);
  fireEvent.click(screen.getByRole('button',{name:'Decrease number of questions'}));expect(input).toHaveValue(27);
  fireEvent.change(input,{target:{value:'50'}});expect(screen.getByRole('button',{name:'Increase number of questions'})).toBeDisabled();
  fireEvent.change(input,{target:{value:'1'}});expect(screen.getByRole('button',{name:'Decrease number of questions'})).toBeDisabled();
});

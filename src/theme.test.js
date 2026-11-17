import { act, fireEvent, render, screen } from "@testing-library/react";
import { ThemeSelector, useThemeMode,setThemePreference,toggleThemePreference } from "./theme";

function ThemePage() {
  const { resolved } = useThemeMode();
  return <><ThemeSelector /><p role="status">{resolved}</p></>;
}

test("follows the device until an explicit appearance is chosen, and remembers that choice", () => {
  localStorage.clear();
  const original = window.matchMedia;
  const listeners = new Set();
  const media = { matches: true, addEventListener: (_type, fn) => listeners.add(fn), removeEventListener: (_type, fn) => listeners.delete(fn) };
  window.matchMedia = () => media;
  try {
    const view = render(<ThemePage />);
    expect(screen.getByRole("status")).toHaveTextContent("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    fireEvent.click(screen.getByRole("combobox", { name: "Appearance" }));
    fireEvent.click(screen.getByRole("option", { name: "Light", exact: true }));
    expect(screen.getByRole("status")).toHaveTextContent("light");
    view.unmount();
    render(<ThemePage />);
    expect(screen.getByRole("status")).toHaveTextContent("light");
    fireEvent.click(screen.getByRole("combobox", { name: "Appearance" }));
    fireEvent.click(screen.getByRole("option", { name: "Use device setting", exact: true }));
    act(() => { media.matches = false; for (const fn of listeners) fn(); });
    expect(screen.getByRole("status")).toHaveTextContent("light");
    act(() => { media.matches = true; for (const fn of listeners) fn(); });
    expect(screen.getByRole("status")).toHaveTextContent("dark");
  } finally { window.matchMedia = original; }
});
test('theme transitions commit the latest rapid choice and reduced motion uses an immediate update',async()=>{
 const previous=window.matchMedia,oldTransition=document.startViewTransition;
 window.matchMedia=query=>({matches:false,addEventListener:()=>{},removeEventListener:()=>{}});
 const callbacks=[],transitions=[];
 document.startViewTransition=jest.fn(callback=>{callbacks.push(callback);const transition={skipTransition:jest.fn(),ready:Promise.resolve(),finished:new Promise(()=>{})};transitions.push(transition);return transition;});
 try{
  render(<ThemePage/>);
  act(()=>setThemePreference('light'));await act(async()=>callbacks.shift()());
  act(()=>toggleThemePreference());act(()=>toggleThemePreference());
  expect(transitions.at(-2).skipTransition).toHaveBeenCalled();
  await act(async()=>{while(callbacks.length)callbacks.shift()();});
  expect(screen.getByRole('status')).toHaveTextContent('light');
  window.matchMedia=query=>({matches:query.includes('reduced-motion'),addEventListener:()=>{},removeEventListener:()=>{}});
  const count=document.startViewTransition.mock.calls.length;act(()=>setThemePreference('dark'));expect(screen.getByRole('status')).toHaveTextContent('dark');expect(document.startViewTransition).toHaveBeenCalledTimes(count);
 }finally{window.matchMedia=previous;if(oldTransition)document.startViewTransition=oldTransition;else delete document.startViewTransition;}
});

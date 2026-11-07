import { act, fireEvent, render, screen } from "@testing-library/react";
import { ThemeSelector, useThemeMode } from "./theme";

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

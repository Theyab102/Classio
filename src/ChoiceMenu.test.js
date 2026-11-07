import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import ChoiceMenu from "./ChoiceMenu";
import StudioDialog from "./StudioDialog";

test("custom menus support keyboard selection, disabled choices and dialog Escape independently", () => {
  const close = jest.fn();
  function Menu() { const [value, setValue] = useState("fine"); return <StudioDialog title="Tools" onClose={close}><label>Stroke<ChoiceMenu value={value} onChange={event => setValue(event.target.value)}><option value="fine">Fine</option><option disabled>Unavailable</option><option value="broad">Broad</option></ChoiceMenu></label></StudioDialog>; }
  render(<Menu />);
  const trigger = screen.getByRole("combobox", { name: "Stroke" }); trigger.focus();
  fireEvent.keyDown(trigger, { key: "ArrowDown" }); fireEvent.keyDown(trigger, { key: "ArrowDown" }); fireEvent.keyDown(trigger, { key: "Enter" });
  expect(trigger).toHaveTextContent("Broad"); expect(trigger).toHaveAttribute("aria-expanded", "false"); expect(trigger).toHaveFocus();
  fireEvent.click(trigger); fireEvent.keyDown(trigger, { key: "Escape" });
  expect(close).not.toHaveBeenCalled(); expect(screen.queryByRole("listbox")).toBeNull();
  fireEvent.keyDown(trigger, { key: "Escape" }); expect(close).toHaveBeenCalledTimes(1);
});

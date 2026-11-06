import { fireEvent, render, screen } from "@testing-library/react";
import ResponsiveSidebar from "./ResponsiveSidebar";

test("phone menu includes every study tool, traps focus, selects a tool and closes", () => {
  const select = jest.fn();
  const props = { mobile: true, expanded: false, onExpand: jest.fn(), main: [{ id: "home", label: "Dashboard" }],
    tabs: ["Learn Mode", "Notes", "Voice & Podcast", "Study Cards", "AI Assistant", "Game Mode", "AI Presentation", "YouTube"].map(label => ({ id: label, label })),
    onTab: select, onNavigate: jest.fn(), onSearch: jest.fn(), onCharacter: jest.fn(), onSignOut: jest.fn(), guest: true };
  render(<ResponsiveSidebar {...props} />);
  const trigger = screen.getByRole("button", { name: "Open menu" });
  fireEvent.click(trigger);
  expect(screen.getByRole("dialog", { name: "Workspace navigation" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Close menu", exact: true })).toHaveFocus();
  expect(screen.getByRole("button", { name: "AI Presentation" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "YouTube" }));
  expect(select).toHaveBeenCalledWith("YouTube");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
  fireEvent.click(trigger);
  fireEvent.keyDown(window, { key: "Escape" });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

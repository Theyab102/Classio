import { fireEvent, render, screen } from "@testing-library/react";
import ResponsiveSidebar from "./ResponsiveSidebar";

test("phone menu includes every study tool, traps focus, selects a tool and closes", () => {
  const select = jest.fn();
  const props = { mobile: true, expanded: false, onExpand: jest.fn(), main: [{ id: "home", label: "Library" }, { id: "assistant", label: "AI Assistant" }],
    tabs: ["Notes", "Review", "Audio", "Slides", "Source", "Link"].map(label => ({ id: label, label })),
    onTab: select, onNavigate: jest.fn(), onSearch: jest.fn(), onCharacter: jest.fn(), onSignOut: jest.fn(), guest: true };
  render(<ResponsiveSidebar {...props} />);
  const trigger = screen.getByRole("button", { name: "Open menu" });
  fireEvent.click(trigger);
  expect(screen.getByRole("dialog", { name: "Workspace navigation" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Close menu", exact: true })).toHaveFocus();
  expect(screen.getByRole("button", { name: "Slides" })).toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: "AI Assistant", exact: true })).toHaveLength(1);
  fireEvent.click(screen.getByRole("button", { name: "Link", exact: true }));
  expect(select).toHaveBeenCalledWith("Link");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
  fireEvent.click(trigger);
  fireEvent.keyDown(window, { key: "Escape" });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

import { fireEvent, render, screen } from "@testing-library/react";
import LibraryHome from "./LibraryHome";
import ActionMenu from "./ActionMenu";
import AvatarEditor from "./AvatarEditor";

test("creation choices share one menu and leave the library free of duplicate actions", () => {
  const createNote = jest.fn();
  render(<LibraryHome folders={[]} createNote={createNote} />);
  expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Create new" }));
  fireEvent.click(screen.getByRole("menuitem", { name: "Build a lesson" }));
  expect(createNote).toHaveBeenCalledWith("lessons");
  expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Create new" })).toHaveFocus();
});

test("file actions retain the source folder and offer a nested move destination", () => {
  const move = jest.fn(), rename = jest.fn();
  const file = { id: "file-a", name: "Physics", type: "text/plain", notes: "Keep my writing" };
  render(<LibraryHome folders={[{ id: "inbox", files: [file] }, { id: "science", name: "Science", files: [] }]}
    move={move} rename={rename} />);
  fireEvent.click(screen.getByRole("button", { name: "Actions for Physics" }));
  fireEvent.click(screen.getByRole("menuitem", { name: "Move to" }));
  fireEvent.click(screen.getByRole("menuitem", { name: "Science" }));
  expect(move).toHaveBeenCalledWith(file, "inbox", "science");
  fireEvent.click(screen.getByRole("button", { name: "Actions for Physics" }));
  fireEvent.click(screen.getByRole("menuitem", { name: "Rename" }));
  expect(rename).toHaveBeenCalledWith(expect.objectContaining({ id: "file-a", scope: "inbox" }));
  expect(file.notes).toBe("Keep my writing");
});

test("an action menu supports keyboard navigation and Escape returns to its trigger", () => {
  render(<ActionMenu label="Actions" items={[{ label: "One" }, { label: "Two" }]} />);
  const trigger = screen.getByRole("button", { name: "Actions" });
  fireEvent.click(trigger);
  fireEvent.scroll(window);
  expect(screen.getByRole("menu")).toBeInTheDocument();
  expect(screen.getByRole("menuitem", { name: "One" })).toHaveFocus();
  fireEvent.keyDown(screen.getByRole("menu"), { key: "End" });
  expect(screen.getByRole("menuitem", { name: "Two" })).toHaveFocus();
  fireEvent.keyDown(screen.getByRole("menu"), { key: "Escape" });
  expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

test("character updates preserve the existing appearance and Escape closes the editor", () => {
  const character = { skin: "#c68642", hair: "#493326", hairStyle: 6, top: "#417f99", name: "Sam" };
  const change = jest.fn(), close = jest.fn();
  render(<AvatarEditor character={character} onChange={change} onClose={close} />);
  fireEvent.click(screen.getByRole("button", { name: "Hair", exact: true }));
  fireEvent.click(screen.getByRole("button", { name: "Hairstyle: Two buns", exact: true }));
  expect(change).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Save changes", exact: true }));
  expect(change).toHaveBeenCalledWith(expect.objectContaining({ ...character, hairStyle: 13 }));
  fireEvent.keyDown(window, { key: "Escape" });
  expect(close).toHaveBeenCalledTimes(2);
});

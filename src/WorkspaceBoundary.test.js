import { render, screen } from "@testing-library/react";
import WorkspaceBoundary from "./WorkspaceBoundary";

test("a failed workspace import offers recovery instead of leaving a blank page", () => {
  const logging = jest.spyOn(console, "error").mockImplementation(() => {});
  function UnavailableChunk() { throw new Error("Loading chunk failed"); }
  try {
    render(<WorkspaceBoundary><UnavailableChunk /></WorkspaceBoundary>);
    expect(screen.getByRole("alert")).toHaveTextContent("workspace could not open");
    expect(screen.getByRole("button", { name: "Reload Classio" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Privacy Policy" })).toHaveAttribute("href", "/privacy");
  } finally { logging.mockRestore(); }
});

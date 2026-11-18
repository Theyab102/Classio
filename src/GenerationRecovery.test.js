import { render, screen, fireEvent } from "@testing-library/react";
import GenerationRecovery from "./GenerationRecovery";
test("an auth failure before the first section never claims generated progress was saved", () => {
  const resume = jest.fn(),
    job = { status: "failed", sections: [], total: 1 };
  render(<GenerationRecovery job={job} onResume={resume} />);
  expect(
    screen.queryByText("Generation progress saved"),
  ).not.toBeInTheDocument();
  expect(screen.queryByText(/0 of 1/)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(resume).toHaveBeenCalledWith(job);
});
test("completed sections and a saved outline keep a real recovery path", () => {
  const page = render(
    <GenerationRecovery
      job={{ status: "failed", sections: [{ title: "Saved" }], total: 20 }}
      onResume={() => {}}
    />,
  );
  expect(screen.getByText(/1 of 20 sections saved/)).toBeInTheDocument();
  page.rerender(
    <GenerationRecovery
      job={{
        status: "failed",
        sections: [],
        outline: [{ title: "Plan" }],
        total: 20,
      }}
      onResume={() => {}}
    />,
  );
  expect(screen.getByText(/Outline saved/)).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Resume generation" }),
  ).toBeInTheDocument();
});

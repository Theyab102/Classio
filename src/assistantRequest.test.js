import { assistantHistory, isSimpleGreeting, materialExcerpt } from "./assistantRequest";
import { parseStudyRequest } from "../server/study-request";

test("image chat uses the server schema and only resends the most recent attachment", () => {
  const messages = [
    { role: "user", content: "First diagram", image: "data:image/png;base64,aGVsbG8=" },
    { role: "assistant", content: "An explanation" },
    { role: "user", content: "Now this diagram", image: "data:image/png;base64,dGVzdA==" },
  ];
  const payload = { messages: assistantHistory(messages, "Use only the selected material."), maxTokens: 1000 };
  expect(parseStudyRequest(payload)?.vision).toBe(true);
  expect(payload.messages[1].content).toBe("First diagram");
  expect(payload.messages[3].content[0].type).toBe("image_url");
});
test("large histories stay inside the actual server input limit", () => {
  const history = Array.from({ length: 60 }, (_, i) => ({ role: i % 2 ? "assistant" : "user", content: "x".repeat(12000) }));
  expect(parseStudyRequest({ messages: assistantHistory(history, "y".repeat(25000)), maxTokens: 1000 })).not.toBeNull();
  expect(JSON.stringify(assistantHistory(history, "Study tutor.")).length).toBeLessThan(14500);
});
test("greetings are separated from study questions and excerpts find later relevant material", () => {
  for (const greeting of ["hi", "Hey!", "hay", "مرحبا", "hello Classio"]) expect(isSimpleGreeting(greeting)).toBe(true);
  expect(isSimpleGreeting("Hi, explain projectile motion")).toBe(false);
  const source = "# Physics\n\n" + Array.from({ length: 50 }, (_, i) => "Unrelated paragraph " + i + ". ".repeat(150)).join("\n\n") + "\n\nDrag opposes the direction of motion. Terminal velocity occurs when drag balances weight.";
  const excerpt = materialExcerpt(source, "Explain terminal velocity", 1400);
  expect(excerpt).toContain("Terminal velocity"); expect(excerpt.length).toBeLessThanOrEqual(1400);
});

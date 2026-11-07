import { assistantHistory } from "./assistantRequest";
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
});

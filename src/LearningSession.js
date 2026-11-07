import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

const Session = createContext({ activity: null, setActivity: () => {}, ask: () => {} });
export function LearningSession({ children }) {
  const [activity, setActivity] = useState(null);
  const [question, setQuestion] = useState(null);
  const ask = useCallback(text => setQuestion({ text, id: Date.now() }), []);
  const value = useMemo(() => ({ activity, setActivity, question, ask, clearQuestion: () => setQuestion(null) }), [activity, question, ask]);
  return <Session.Provider value={value}>{children}</Session.Provider>;
}
export const useLearningSession = () => useContext(Session);
export function useStudyActivity(fileId, type, text, title = "") {
  const { setActivity } = useLearningSession();
  useEffect(() => {
    if (!fileId) return;
    const value = { fileId, type, text: String(text || "").slice(0, 9000), title };
    setActivity(value);
    return () => setActivity(current => current === value ? null : current);
  }, [fileId, type, text, title, setActivity]);
}

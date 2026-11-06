import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import Entry from "./Entry";
import "./theme";
import "./locale";
import "./viewport";
import "./workspace.css";

createRoot(document.getElementById("root")).render(
  <StrictMode><Entry /></StrictMode>
);

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/dm-sans";
import "@fontsource-variable/fraunces";
import "./index.css";
import Entry from "./Entry";
import "./theme";
import "./workspace.css";

createRoot(document.getElementById("root")).render(
  <StrictMode><Entry /></StrictMode>
);

import { Component } from "react";

export default class WorkspaceBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }

  render() {
    if (!this.state.failed) return this.props.children;
    return <main className="workspace-recovery" role="alert">
      <img src="/ClassioLogo-transparent.png" width="40" height="40" alt="" />
      <h1>The workspace could not open</h1>
      <p>A file may have failed to download. Check your connection, then reload Classio.</p>
      <p>Reloading ends an open guest session. It does not clear your saved browser files.</p>
      <button className="entry-button" onClick={() => window.location.reload()}>Reload Classio</button>
      <a href="/privacy">Privacy Policy</a>
    </main>;
  }
}

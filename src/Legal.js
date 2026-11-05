const operator = process.env.REACT_APP_CLASSIO_OPERATOR || "Classio";
const contactEmail = process.env.REACT_APP_CLASSIO_CONTACT_EMAIL;
const copyrightEmail = process.env.REACT_APP_CLASSIO_COPYRIGHT_EMAIL || contactEmail;
const copyrightAgent = process.env.REACT_APP_CLASSIO_COPYRIGHT_AGENT;
const copyrightAddress = process.env.REACT_APP_CLASSIO_COPYRIGHT_ADDRESS;
const copyrightPhone = process.env.REACT_APP_CLASSIO_COPYRIGHT_PHONE;

function Contact({ email = contactEmail }) {
  return email
    ? <a href={`mailto:${email}`}>{email}</a>
    : <span>Contact details are pending public launch.</span>;
}

export const legalPages = {
  "/privacy": {
    title: "Privacy Policy",
    content: (
      <>
        <p>This policy explains how {operator} handles information when you visit Classio or use its study tools.</p>
        <h2>Eligibility</h2>
        <p>Classio is intended for people aged 13 or older. The age check runs in your browser. Classio keeps the eligibility result and your terms acknowledgment in the current browser tab. It does not send or store your date of birth.</p>
        <h2>Accounts and study materials</h2>
        <p>Google sign-in and Firebase process your account identifier and basic Google profile details. If you choose to sign in, Firebase may store your synced folder information, progress and study content. Guest work is kept for the open workspace session. Where group features are available, Firebase also stores shared group activity. Classio keeps original personal file uploads and some preferences in your browser. Files and messages shared with a study group can be accessed by people in the group.</p>
        <h2>Optional study tools</h2>
        <p>When you use an AI tool, Classio sends the prompt, relevant study text, chat context, selected image or selected audio needed for the request to Groq through Classio’s server. To enforce short-lived AI request limits, the server stores a keyed hash of your connection’s IP address alongside a request count. Some tools also contact Wikimedia, YouTube, YouTubeTranscript or Pollinations to find media, read public links or generate an image you request. Live speech may use your browser’s speech provider. Downloading podcast audio sends the script to Classio’s Piper service on Hugging Face. Microphone and screen sharing require your browser’s permission.</p>
        <h2>Hosting and tracking</h2>
        <p>Notes and presentation image searches send search terms to Wikimedia Commons. Displaying those images can contact Wikimedia’s image servers. Opening the separate Google Images search sends the search term to Google. When you view a PDF or extract a figure from it, Classio processes the PDF in your browser. Using AI on the PDF can send selected study text to the AI provider as described above.</p>
        <p>The hosting and authentication providers may process connection and security logs. Classio uses browser storage for features such as sign-in, settings and saved work. This version does not include ad tracking, Google Analytics or session replay. Its fonts are served with the app.</p>
        <h2>Retention and privacy requests</h2>
        <p>Browser copies remain until you delete the relevant material, clear site data or use the app's local cache control where applicable. Account and group records can remain after sign-out. For access, correction, deletion, retention questions or to report an account for a child under 13, contact <Contact />.</p>
      </>
    ),
  },
  "/terms": {
    title: "Terms and Conditions",
    content: (
      <>
        <p>These terms govern use of Classio, operated by {operator}. You must be at least 13 to use the workspace. If you are a minor, use it only with any permission required where you live.</p>
        <h2>Your materials</h2>
        <p>You retain your rights in your materials. By uploading or sharing them, you give Classio permission to process, store and display them as needed to provide the features you use. Upload or share material only if you own it or have permission or another legal right to use it. Keep private information out of shared study rooms and AI prompts.</p>
        <h2>Use of the workspace</h2>
        <p>Use Classio lawfully and respect other users' privacy and rights. Do not share infringing or harmful material, impersonate others, disrupt the service or try to access someone else's files. Classio may remove material or restrict access when needed to address abuse or comply with the law.</p>
        <h2>Study tools and billing</h2>
        <p>Check generated notes, answers and citations against your course material. Classio is free to use. Free AI processing is subject to shared service limits. Signing in is optional and lets you save and sync your study progress.</p>
        <h2>Contact</h2>
        <p>For questions about these terms, contact <Contact />. To report infringement, see the <a href="/copyright">copyright notice</a>.</p>
      </>
    ),
  },
  "/copyright": {
    title: "Copyright",
    content: (
      <>
        <p>Only upload or share materials you have the right to use. For an infringement report, contact <Contact email={copyrightEmail} />.</p>
        {copyrightAgent && copyrightAddress && copyrightPhone && copyrightEmail && (
          <address>
            <strong>{copyrightAgent}</strong><br />
            {copyrightAddress}<br />
            {copyrightPhone}<br />
            <Contact email={copyrightEmail} />
          </address>
        )}
        <h2>What to include</h2>
        <ul>
          <li>Your physical or electronic signature and contact information.</li>
          <li>The work you own or represent and the material you want removed, with enough detail for us to find it.</li>
          <li>A statement that you have a good-faith belief the use is unauthorized.</li>
          <li>A statement that the notice is accurate and, under penalty of perjury, that you are authorized to act for the owner.</li>
        </ul>
        <p>If your material was removed in error, you can use the same contact to ask about submitting a counter-notice. Classio may restrict accounts that repeatedly infringe copyright in appropriate circumstances.</p>
      </>
    ),
  },
};

export function LegalLinks() {
  return (
    <footer className="legal-links" aria-label="Policies">
      <a href="/privacy">Privacy Policy</a>
      <a href="/terms">Terms and Conditions</a>
      <a href="/copyright">Copyright</a>
    </footer>
  );
}

export function LegalPage({ page }) {
  return (
    <div className="entry-shell">
      <header className="entry-header"><a className="entry-brand" href="/"><img src="/ClassioLogo.png" width="32" height="32" alt="" />Classio</a></header>
      <main className="legal-page">
        <h1>{page.title}</h1>
        {page.content}
      </main>
      <LegalLinks />
    </div>
  );
}

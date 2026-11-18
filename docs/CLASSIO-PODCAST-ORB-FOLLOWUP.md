# Inline podcast questions and visible playback controls

Continues from `0eb5eed9f0c2213ef69a19d94d4fc5f5f322e648`.

The playback controls previously rendered only when an audio blob was available.
A saved transcript with missing/unprepared audio therefore showed no play, seek,
skip or volume controls. The transport now stays visible for that episode.
**Play** prepares its existing transcript and starts actual audio when ready;
**Prepare audio** remains available as an alternative without autoplay. Failures
keep the transcript and controls available with an error. No fake audio player
is created for a transcript alone.

Play/pause, back/forward 15 seconds, the seek bar, playback speed, volume and mute
use the real media element. Seeking uses current media time and clamps at the
start/end. Volume and mute also apply to a spoken host answer. Episode seeking
and playback pause during a voice question so the two audio streams do not
overlap; the controls remain visible.

## Voice interaction

The Ask button now owns the voice flow inline, immediately below the host
characters. It remains the same button while its width and corner radius animate
into a 68-pixel orb. A rotating ring and audio-level response indicate listening;
status text distinguishes permission, recording, transcription, thinking and
speaking. Reduced motion removes the transition and spinning while preserving
the states and keyboard-operable controls.

- Clicking Ask pauses the episode and requests the microphone.
- Clicking the orb again sends the captured question. A three-second pause after
  sustained speech sends automatically, within the requested two-to-five-second
  range. Initial silence/brief noise is not submitted as a question.
- Cancel stops capture and aborts pending transcription, generation or speech.
  Each async continuation checks its owning job, so a late permission grant,
  transcript or answer cannot restart a cancelled flow.
- The host replies aloud. Clicking the orb during a reply interrupts it and
  starts a follow-up. No second dialog, transcription textarea or listening menu
  opens. Open text tutor remains the explicit text alternative.
- Completion or cancellation restores the original position and resumes the
  episode if it was playing beforehand. An already paused episode stays paused.
  **Resume podcast** explicitly ends the question and starts the episode.
  Asking while saved audio metadata is loading retains the saved position.

Microphone capture uses browser echo cancellation/noise suppression, is bounded
to 45 seconds and retains the existing audio byte limit. Silence detection is a
local energy heuristic, so microphone/device quality and background sound still
matter. A denied or unavailable microphone shows an inline retry explanation.
Provider errors do not fabricate a host answer or a playable episode.

## Verification

```sh
CI=true npm test -- --watchAll=false --maxWorkers=2
npm run test:server
npm run build
npm run test:podcast
CLASSIO_SESSION_TARGET=phone npm run test:sessions
CLASSIO_WORKFLOW_TARGET=phone npm run test:workflows
npm run test:browser
```

Frontend results: **43 suites / 184 tests** pass, including manual send,
duplicate-click prevention, silence intervals, cancellation, late permission and
transcription, speech retry, autoplay intent, seek bounds, mute/volume and a saved
position before metadata loads. All **42 API tests** pass. The production build
passes with the remaining legacy lint warnings.

The dedicated Chromium matrix covers 1440×1000 desktop, 1024×768 landscape tablet,
768×1024 portrait tablet and 390×844 phone, plus both themes and reduced motion.
It exercises real MediaRecorder output, Web Audio analysis of generated audio
followed by silence, native WAV playback, restored transport, skip/volume/mute,
same-button morphing, cancellation, late responses and spoken interruption. The
normal phone session/recording suite and phone workflow suite also pass.
The broad desktop/phone study suite passes as well, including responsive checks
at 320/768/1024/1280 pixels and 125%/150% zoom emulation.

Provider HTTP responses are explicit fixtures. The checks do not verify live
provider credentials, physical microphone quality, mobile Safari or production
deployment activation. No Firebase, AI-provider or speech-service settings were
changed. No speech-server source changes are required.

Logs: `/workspace/.onboarding/orb-*.log`. Screenshots:
`/workspace/artifacts/classio/podcast-orb/`. Test fixtures and screenshots are not
shipped in browser assets.

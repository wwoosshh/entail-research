# Rating codebook: the kind of defect behind a reported wrong output, and whether a caller could have seen it

You will receive a list of GitHub issues from local LLM runtimes (a local model server, a desktop app that runs
models, an inference library). Each issue reports a wrong or degraded model output. For each issue, answer two
questions, judging only from the issue's text and, when given, the description of the fix. Do not guess beyond the
text.

## Question K: the kind of defect (choose exactly one)

- **K1** — A property of a value (its storage format, its numeric unit or scale, the reference point of a position,
  the range of positions that are valid, the moment it was produced, which item it stands for, or which of several
  declarations takes precedence) is set or declared in one component, and another component did not receive it,
  ignored it, read it under a different assumption, or substituted a default. The defect is in the hand-over
  between components, not in the arithmetic of either one.
- **K2** — Arithmetic or numerical error inside one component: a wrong formula, a precision or overflow problem, a
  NaN or infinity produced by a computation, an accumulation in the wrong order or type.
- **K3** — Memory or object lifetime: a buffer freed, reused or overwritten while still in use, a weak or dangling
  reference, a captured graph or cached structure holding an address that no longer belongs to it.
- **K4** — Control-flow, dispatch or parsing logic inside one component: a wrong branch, a wrong path chosen for
  an input, a parser that reads a stream incorrectly, an off-by-one in a loop, a missing case.
- **K5** — A defect specific to hardware, a driver, a platform or a compiler backend, where the same code is correct
  elsewhere.
- **K6** — The reporter's configuration or usage was wrong; the software behaved as designed.
- **K7** — The text does not say enough to decide (no cause identified, no fix described, contradictory reports).

Rules for K:
- Judge the ROOT defect the text identifies, not the symptom. A NaN caused by a missing scale hand-over is K1; a
  NaN from a formula is K2.
- If a fix is described, weigh it most: what did the fix change?
- If two categories seem to apply, choose the one the fix (or the reporter's analysis) points to; if you truly
  cannot choose, pick K7 and say why.

## Question V: could a program calling the runtime through its API have noticed the problem?

Imagine a program that sends requests to this runtime over its API and uses the answers. At the time of the
problem, the program has only:
(a) the request it sent;
(b) the response it got back - the text, and response fields such as token counts, the reason generation stopped,
    timings;
(c) what the runtime's API reports about the loaded model and its settings (for example a model-information or
    list-loaded-models endpoint);
(d) the metadata stored in the model file itself.

- **V1** — Yes: from one request and the items above, a program could have noticed that something was wrong.
- **V2** — Only by comparing: the program would have had to run the runtime twice (different settings, backends or
  paths) and compare the results.
- **V3** — No: the problem could not be noticed from outside the runtime.
- **V4** — The text does not say enough to decide.

Give a confidence from 1 (guess) to 3 (the text is explicit) for each answer, and one sentence of evidence.

Output format, one line per issue, exactly:
`<issue url> | <K1..K7> | <K confidence 1-3> | <V1..V4> | <V confidence 1-3> | <one-sentence evidence>`

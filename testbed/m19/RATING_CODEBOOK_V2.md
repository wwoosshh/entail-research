# Rating codebook v2: the kind of defect behind a reported wrong output, and where the information lives

You will receive a list of GitHub issues from AI inference engines (an LLM server, a model library, an image
pipeline). Each issue reports a wrong or degraded model output. For each issue, decide which ONE category below best
describes the defect that produced the wrong output, judging only from the issue's text and, when given, the
description of the fix. Do not guess beyond the text: if the text does not say enough, choose K7.

Categories (choose exactly one):

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

Rules:
- Judge the ROOT defect the text identifies, not the symptom. A NaN caused by a missing scale hand-over is K1; a
  NaN from a formula is K2.
- If a fix is described, weigh it most: what did the fix change?
- If two categories seem to apply, choose the one the fix (or the reporter's analysis) points to; if you truly
  cannot choose, pick K7 and say why.
- Give a confidence from 1 (guess) to 3 (the text is explicit), and one sentence quoting or paraphrasing the
  evidence.

Second question (answer it for every issue as well): where does the wrong or lost information live?

- **D** — in the data the program computes on or moves, or in how it computes: a property of numeric arrays (memory
  layout, axis order, strides or contiguity, element type, numeric scale factors or quantization grouping, which
  row, slot, page, head or expert an entry belongs to, position indices, whether a cached entry is the right one or
  still valid, when a buffer may be reused or freed); a numeric computation (formula, precision, overflow); the
  choice of compute routine or backend; the order of work between streams or devices, or communication between
  devices.
- **I** — in what surrounds the computation: text, the token ids a tokenizer produces, prompt templates, fields of a
  request or response, configuration values read at startup (other than the ones listed under D), parsing of the
  model's output text, or the caller's own inputs.
- **U** — the text does not let you decide.

The two questions are independent: answer each from the text on its own terms.

Output format, one line per issue, exactly:
`<issue url> | <K1..K7> | <D, I or U> | <confidence 1-3> | <one-sentence evidence>`

# Rating rules

You will rate short descriptions of software defects in machine-learning inference and image-generation software
(vLLM, SGLang, Hugging Face transformers, diffusers). Each item says what went wrong and, usually, what the fix
changed. Answer three questions per item. There are no expected answers; rate what the text says.

## Q1. Where does the wrong or lost information live?

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

## Q2. Which check would have shown that something is wrong?

Assume the check runs with the configuration and input that trigger the defect. Choose every letter that applies.

- **M** — run the same input under two execution settings that are meant to give the same result, and compare the
  outputs. Examples of such pairs: graph capture on / off, compiled / not compiled, a cache on / off, speculative
  decoding on / off, several devices / one device, disaggregated / colocated serving, a request alone / in a batch,
  a different chunk or block size, a fused / unfused path, another backend or platform.
- **R** — compare with an independent reference implementation of the same model or operation on the same input.
- **T** — compare a property that the array records about itself (element type, contiguity, strides, shape) with
  what the routine that consumes it requires.
- **L** — compare the loaded model parameters with the contents of the checkpoint.
- **F** — check that values stay finite (no NaN or infinity).
- **N** — none of these would show it.

## Q3. Could the defect be triggered on one consumer NVIDIA GPU?

A single RTX 4070 Ti (12 GB, compute capability 8.9) with CUDA, using a small model of the same architecture if the
original model is large.

- **Y** — yes.
- **N** — no: it needs several GPUs, a different GPU generation or vendor (Hopper- or Blackwell-only kernels, AMD
  ROCm, Intel XPU, Apple MPS, other accelerators), or resources far beyond one such GPU.
- **U** — the text does not let you decide.

## Output

One line per item, in the order of the packet, nothing else:

    X001 | D | M R | Y | 3 | reason in at most 25 words

The fields are: id, Q1, Q2 letters separated by spaces, Q3, confidence (1 = unsure, 3 = sure), reason.

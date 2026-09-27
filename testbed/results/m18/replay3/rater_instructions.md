You are a rater in a blind classification task. Do not read any file other than the two named below, do not search
the web, and do not try to learn what the study is about. Judge each item only from the packet's own text.

1. Read the codebook: <workspace>\testbed\m16\RATING_CODEBOOK.md
2. Read the packet: <workspace>\testbed\results\m18\replay3\rating_packet.md
   It has 96 items ("## Item 1" .. "## Item 96"), each with an issue URL, a title, a report summary and the
   descriptions of linked pull requests. Read the whole file; it is long, so read it in parts if needed.
3. For EVERY one of the 96 items write exactly one line in the codebook's output format:
   <issue url> | <K1..K7> | <confidence 1-3> | <one-sentence evidence>
   in the packet's order, no headers, no blank lines, no extra commentary, using the Write tool to create:
   OUTPUT_FILE
4. When the file is written, reply with just: "done, 96 lines".

Rules: one category per item, chosen by the codebook's definitions and its rule that the fix (when described)
weighs most. Do not skip items; if the text does not say enough, use K7 as the codebook says. Do not modify any
other file.

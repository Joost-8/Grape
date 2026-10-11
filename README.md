# FineNotes handwriting model

An open, on-device handwriting recogniser for pen and stylus ink: it reads
lines of handwriting from the pen's strokes (not from an image) and returns
text. It is the model behind the on-device reading in
[FineNotes](https://github.com/Joost-8/FineNotes), the handwriting notebook
plugin for Obsidian.

| | |
|---|---|
| File | [`finenotes-hwr-1.0.bin`](finenotes-hwr-1.0.bin), 3.50 MB |
| SHA-256 | `87a9ddd56b6bcf01e66a027ff74f1b8ba9a58fe0d4b0077e6abb6455710d21f9` |
| Network | 5-layer bidirectional LSTM, 128 units per direction, CTC, int8 weights (1.74 M parameters) |
| Language model | 6-gram character model in the same file |
| Language | English: Latin letters, digits and common symbols (95 characters) |
| Licence | Apache 2.0, with the attributions in [`NOTICE`](NOTICE) |
| Status | Beta |

## What is here

- [`finenotes-hwr-1.0.bin`](finenotes-hwr-1.0.bin): the model, network and
  language model in one file.
- [`MODEL-CARD.md`](MODEL-CARD.md): what it does, what it is for and not for,
  how it was trained, and on what data.
- [`eval.md`](eval.md): its accuracy on held-out writers from every source,
  measured with this file.
- [`FORMAT.md`](FORMAT.md): the file format, enough to write a reader in any
  language.
- [`NOTICE`](NOTICE): the datasets it was trained on, their authors and the
  terms they were used under. It must travel with every copy of the model.

Only the model is published here. The code that runs it lives in FineNotes
(`src/recognition/hwr/`).

## Accuracy, in short

Character error rate with the language model, on writers the model never
saw: 1.7 % to 4.6 % across the public test sets (IAM-OnDB, UNIPEN,
DeepWriting, ImadocSen, BRUSH). On one new writer's lecture-note lines on an
iPad it is 14 %, and 3.4 % after adapting to that writer. Details and every
test set are in [`eval.md`](eval.md).

## Training data

Only data whose owners allow an open, commercially usable model: BRUSH and
IAM-OnDB (with their authors' written permission), UNIPEN, PluColl and
UJIpenchars2, plus synthetic lines built from their writers' ink. No user's
handwriting is in the model. The full list, with credits and the changes made
to each source, is in [`NOTICE`](NOTICE) and [`MODEL-CARD.md`](MODEL-CARD.md).

## Licence

Copyright 2026 Joost (FineNotes). Licensed under the Apache License, Version
2.0: see [`LICENSE`](LICENSE). The attributions in [`NOTICE`](NOTICE) are part
of the licence terms and must be kept with the model.

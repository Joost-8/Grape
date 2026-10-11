# Grape

An open, on-device handwriting recogniser for pen and stylus ink. It reads
lines of handwriting from the pen's strokes (not from an image) and returns
text. Grape is the model behind the on-device reading in
[FineNotes](https://github.com/Joost-8/FineNotes), the handwriting notebook
plugin for Obsidian.

> **Work in progress.** Grape reads well on public test sets, but on real
> notes written on an iPad it still gets about one character in seven
> wrong. The limit is data, not the network: there is far too little real
> stylus handwriting, from enough different writers, to train on. More real
> data is what will improve it most.

| | |
|---|---|
| File | [`grape-1.0.bin`](grape-1.0.bin), 3.50 MB (internal name `finenotes-hwr-1.0`) |
| SHA-256 | `87a9ddd56b6bcf01e66a027ff74f1b8ba9a58fe0d4b0077e6abb6455710d21f9` |
| Network | 5-layer bidirectional LSTM, 128 units per direction, CTC, int8 weights (1.74 M parameters) |
| Language model | 6-gram character model in the same file |
| Language | English: Latin letters, digits and common symbols (95 characters) |
| Licence | Apache 2.0, with the attributions in [`NOTICE`](NOTICE) |
| Status | Work in progress |

## What is here

- [`grape-1.0.bin`](grape-1.0.bin): the model, network and language model in
  one file.
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
DeepWriting, ImadocSen, BRUSH). On one new writer's handwriting on an iPad it
is 14 % (copied lines) to 19 % (real lecture notes). Details and every test
set are in [`eval.md`](eval.md).

## Real data is the main constraint

The gap between the public test sets and real iPad notes is the gap between
the data Grape learned from and the ink it meets. Its training data is old
pen-tablet and whiteboard handwriting, plus synthetic lines built from it:
there is very little real stylus handwriting, on modern tablets, from many
writers, that may be used for an open model. Measurements so far point the
same way: a larger network, more synthetic lines and distillation from a
bigger model each moved the real-notes numbers by a point at most, while
real lines from the writer moved them by ten.

## Fine-tuning works very well

Adapting Grape to one writer is cheap and very effective. On that writer's
copy sheet, 300 steps of tuning on their other lines take the character
error rate from 14.2 % to 3.4 % (cross-validated), and the gain carries over
to text the tuning never saw. A tuning run takes under half a minute on a
GPU and a few minutes on a laptop CPU.

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

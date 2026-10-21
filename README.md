# Grape

An open, on-device handwriting recogniser for pen and stylus ink. It reads
lines of handwriting from the pen's strokes (not from an image) and returns
text. Grape is the model behind the on-device reading in
[FineNotes](https://github.com/Joost-8/FineNotes), the handwriting notebook
plugin for Obsidian.

> **Work in progress.** Grape reads well on public test sets, but on real
> notes written on an iPad it still gets about one character in eight
> wrong. The limit is data, not the network: there is far too little real
> stylus handwriting, from enough different writers, to train on. More real
> data is what will improve it most.

| | |
|---|---|
| File | [`grape-2.0.bin`](grape-2.0.bin), 5.69 MB |
| SHA-256 | `f553c71367e8881f2918c9a34a44b3f215a54e86b39d6547ff885aede1fa6d04` |
| Network | 5-layer bidirectional LSTM, 192 units per direction, reading Bézier curves, CTC, int8 weights (3.9 M parameters) |
| Language model | 6-gram character model in the same file |
| Language | English: Latin letters, digits and common symbols (95 characters) |
| Licence | Apache 2.0, with the attributions in [`NOTICE`](NOTICE) |
| Status | Work in progress |

## What is here

- [`grape-2.0.bin`](grape-2.0.bin): the model, network and language model in
  one file.
- [`MODEL-CARD.md`](MODEL-CARD.md): what it does, what it is for and not for,
  how it was trained, and on what data.
- [`eval.md`](eval.md): its accuracy on held-out writers from every source,
  measured with this file.
- [`FORMAT.md`](FORMAT.md): the file format, enough to write a reader in any
  language.
- [`NOTICE`](NOTICE): the datasets it was trained on, their authors and the
  terms they were used under. It must travel with every copy of the model.

- [`reader.py`](reader.py): a plain-Python reader (numpy only) for model
  files of format 1, such as Grape 1.0 (the v1.0 release). Grape 2.0 is
  format 2, which also reads Bézier curves: FineNotes' reader
  (`src/recognition/hwr/`, `curves.ts`) is the reference for it.

FineNotes' own reader is TypeScript (`src/recognition/hwr/`).

## Use it

```bash
pip install numpy
python reader.py grape-1.0.bin line.json
```

`line.json` holds one line of handwriting:

```json
{"strokes": [{"points": [[12, 40], [13, 38], [15, 37]], "t0": 1712345678901},
             {"points": [[30, 41], [31, 35]], "t0": 1712345679120}]}
```

Points are `[x, y]` in any unit, with y pointing down as on a screen; the
line is normalised by its own size. `t0` (optional) is when the stroke
started, in milliseconds: with start times the strokes are read in writing
order, without them left to right. From Python:

```python
import reader
model = reader.load("grape-1.0.bin")
print(model.read(strokes, t0s))   # strokes: a list of [(x, y), ...] per stroke
```

## Accuracy, in short

Character error rate with the language model, on writers the model never
saw: 1.0 % to 5.5 % across the public test sets (IAM-OnDB, UNIPEN,
DeepWriting, ImadocSen, BRUSH). On one writer's handwriting on an iPad it is
10 % (copied lines) to 13 % (real lecture notes). Details and every test set
are in [`eval.md`](eval.md).

## Real data is the main constraint

The gap between the public test sets and real iPad notes is the gap between
the data Grape learned from and the ink it meets. Its training data is old
pen-tablet and whiteboard handwriting, plus synthetic lines built from it:
there is very little real stylus handwriting, on modern tablets, from many
writers, that may be used for an open model. Measurements so far point the
same way: Grape 2.0's cleaner labels, wider network and formula training
took the developer's real notes from 21 % to 13 % of characters wrong, and
real lines from the writer took them to 7 %.

## Fine-tuning works very well

Adapting Grape to one writer is cheap and very effective. On that writer's
copy sheet, 300 steps of tuning on their other lines take the character
error rate from 9.7 % to about 5 % (cross-validated), and the gain carries over
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

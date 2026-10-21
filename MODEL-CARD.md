# Grape 2.0, the FineNotes handwriting recogniser

**Status: work in progress.** Real handwriting data is what limits it most; see the README.

| | |
|---|---|
| File | `grape-2.0.bin` (5.69 MB) |
| SHA-256 | `f553c71367e8881f2918c9a34a44b3f215a54e86b39d6547ff885aede1fa6d04` |
| Licence | Apache 2.0 (`LICENSE`); attributions in `NOTICE`, which must travel with the file |
| Language | English (Latin letters, digits and common symbols; 95 characters) |
| File format | version 2 (see `FORMAT.md`) |

## What it does

It reads lines of handwriting written with a pen or an Apple Pencil, from the
pen's strokes (not from an image), and returns text. It runs on the device,
inside FineNotes; no ink or text leaves the device.

- **Network:** a 5-layer bidirectional LSTM with 192 units per direction
  (3.9 million parameters, stored as 8-bit integers), trained with CTC.
- **Input:** the ink of one line, normalised by the height of its
  lower-case letters, resampled, and fitted with cubic Bézier curves
  (tolerance 0.01 of the letter height). The network reads the curves'
  control points, about four times fewer steps than the points they replace,
  which is why it is both larger and faster than Grape 1.0.
- **Language model:** a 6-gram character model (Witten-Bell), stored in the
  same file and used during beam search (beam 16, weight 0.3). It is the
  same language model as Grape 1.0's.
- **Training:** text and maths were learned together. A second output head
  read synthetic formulas during training; it is not in the file, but it
  taught the network the short items, symbols and layouts that notes are
  full of.

## Intended use

- Turning handwritten lines in FineNotes notebooks into searchable text.
- English prose and the kind of short lines found in lecture notes: dates,
  numbers, lists and simple formulas written on one line (`x^2 + 1`,
  `a/b`, `sqrt(x)`).

Out of scope:

- Mathematical layout (fractions, exponents written above the line,
  matrices), tables, diagrams and drawings. Lines like these come out as
  wrong text.
- Other scripts, and accented letters (é, ü, ß are not in the alphabet).
- Identifying or profiling writers.

## Accuracy

Character error rate (CER) in %, measured with this file (int8 network, its
own language model and decoder settings) through FineNotes' own reader. The
public sets are written by people the model never saw in training. Grape 1.0
was measured the same way.

| Test set | Grape 1.0 | Grape 2.0 | What it is |
|---|---|---|---|
| UNIPEN | 2.5 | **2.4** | held-out writers, US tablets (1990s) |
| DeepWriting | **4.5** | 5.5 | held-out IAM-OnDB and ETH tablet writers |
| ImadocSen | 3.2 | **1.8** | French tablet-PC writers; in no training source |
| IAM-OnDB | 1.5 | 1.6 | held-out writers, whiteboard |
| BRUSH | 2.6 | **2.0** | held-out writers |
| Note-style text | 1.2 | **1.0** | held-out BRUSH writers on held-out dates, maths and numbers (synthetic lines) |
| **iPad, copy sheet** | 14.0 | **9.7** | the developer's handwriting, 49 lines copied in FineNotes |
| **iPad, real notes** | 21.2 | **13.4** | the developer's own lecture-note lines, 39 lines |
| iPad, maths sheet | 27.2 | **24.6** | 10 lines of formulas written on one line |
| iPad, another writer | 32.0 | **28.1** | 148 lines by a second FineNotes writer |

Grape 2.0 reads a line in about 24 ms against 34 ms (laptop, features and
network), and the file is 5.7 MB against 3.5 MB.

**Adapting to the writer works very well.** Tuned on the developer's copy
sheet alone (300 steps), the same file reads his notes at 6.9 % and the
maths sheet at 16.1 %, lines it never trained on. On the second writer it
reads 27.6 %: a tuned model stays personal.

**What to expect.** The public test sets are pen tablets and whiteboards.
On FineNotes ink the only measurements are two writers, and the small
iPad sets move by several points between training runs, so treat them as a
rough guide.

## Training data

Every source is cleared for an Apache 2.0 model (see `NOTICE`). No data
under a non-commercial licence was used, and checkpoints were chosen on
held-out UNIPEN writers only. The labels of every handwriting source were
checked line by line against the ink and lines that did not match were
removed (about 1 % of lines), and a numbering fault in some UNIPEN files was
corrected, which recovered about 15,700 lines.

| Source | What it is |
|---|---|
| BRUSH | 162 writers, short English sentences |
| IAM-OnDB | 173 writers, whiteboard lines |
| UNIPEN | 73,569 words and lines, after the correction |
| UNIPEN, joined words and characters | words joined into lines; character lines |
| PluColl | 32 writers' words, single and joined |
| UJIpenchars2 | 60 writers' characters, spliced into lines |
| Synthetic lines, BRUSH and IAM-OnDB ink | real ink cut into letters and words and spliced into new text: sentences, dates, maths, numbers, notes; some in derived writing styles |
| Synthetic formulas | symbols from HWRT and letters from BRUSH and IAM-OnDB writers laid out as formulas from arXiv abstracts and a grammar |

Text for synthetic lines and the language model: Common Voice (CC0), arXiv
titles and abstracts (CC0), Faker (MIT), the DeepMind Mathematics Dataset
generator (Apache 2.0), GSM8K (MIT), FineNotes templates and the developer's
own typed notes.

## Training

48 epochs of 157,000 lines, Adam (peak learning rate 7 × 10⁻⁴), CTC loss for
text and for the formula head, augmentation by rotation, shear, scale and
pen speed. The best epoch on held-out UNIPEN writers was kept.

## Known limitations

- **Digits.** Common confusions: 1 and 7, 6 and b, 1 and l.
- **Formulas.** It reads maths written on one line, not maths with layout.
  Real handwritten formulas are what it lacks most; synthetic ones taught it
  symbols, not how people lay them out.
- **Other writers.** The iPad rows are two writers. Feedback and opt-in
  samples from more writers are what the next version needs most.
- **Writing styles far from the training writers** (very cursive, very
  small, left-handed slant) are likely to read worse; this is not
  measured.
- **Synthetic lines** make up about half of training. They are spliced from
  real ink but can carry artefacts, such as joins between letters that a
  person would not write.

## Version history

- **2.0** (this file): a wider network on Bézier curves, trained on
  label-checked data and with formulas. The 1.0 file stays available in the
  v1.0 release.
- **1.0:** first public model.

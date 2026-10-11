# FineNotes handwriting recogniser 1.0

**Status: beta.**

| | |
|---|---|
| File | `finenotes-hwr-1.0.bin` (3.50 MB) |
| SHA-256 | `87a9ddd56b6bcf01e66a027ff74f1b8ba9a58fe0d4b0077e6abb6455710d21f9` |
| Licence | Apache 2.0 (`LICENSE`); attributions in `NOTICE`, which must travel with the file |
| Trained model | `ship-big2`, 2026-10-09; exported 2026-10-11 |
| Language | English (Latin letters, digits and common symbols; 95 characters) |

## What it does

It reads lines of handwriting written with a pen or an Apple Pencil, from the
pen's strokes (not from an image), and returns text. It runs on the device,
inside FineNotes; no ink or text leaves the device.

- **Network:** a 5-layer bidirectional LSTM with 128 units per direction
  (1.74 million parameters, stored as 8-bit integers), trained with CTC.
- **Input:** the ink of one line, normalised by the height of its
  lower-case letters, resampled, with the timing FineNotes stores (one start
  time per stroke, constant pen speed inside a stroke).
- **Language model:** a 6-gram character model (Witten-Bell), stored in the
  same file and used during beam search (beam 16, weight 0.3); contexts seen fewer than 32 times are pruned, which keeps the file at 3.5 MB. It was
  trained on 400,000 lines of text: 60 % Common Voice sentences, 40 %
  dates, times, numbers, maths, code and note-style lines (see Training
  data).

## Intended use

- Turning handwritten lines in FineNotes notebooks into searchable text.
- English prose and the kind of short lines found in lecture notes: dates,
  numbers, simple formulas, lists.

Out of scope:

- Mathematical layout (fractions, exponents written above the line,
  matrices), tables, diagrams and drawings. Lines like these come out as
  wrong text.
- Other scripts, and accented letters (é, ü, ß are not in the alphabet).
- Identifying or profiling writers.

## Accuracy

Character error rate (CER) and word error rate (WER) in %, measured with
this file (int8 network, its own language model and decoder settings).
Every test set is written by people the model never saw in training.

| Test set | Lines | CER | WER | What it is |
|---|---|---|---|---|
| UNIPEN | 600 | 3.4 | 12.7 | held-out writers, US tablets (1990s) |
| DeepWriting | 600 | 4.6 | 18.5 | held-out IAM-OnDB and ETH writers |
| ImadocSen | 600 | 2.9 | 7.3 | French tablet-PC writers; in no training source |
| IAM-OnDB | 600 | 1.7 | 6.8 | held-out writers, whiteboard |
| BRUSH | 600 | 2.4 | 14.0 | held-out writers |
| UJIpenchars2 | 600 | 0.8 | 3.2 | held-out writers, isolated characters |
| Note-style text | 600 | 1.2 | 5.7 | held-out BRUSH writers on held-out dates, maths and numbers (synthetic lines) |
| **iPad, copy sheet** | 49 | **14.2** | 44.7 | the developer's handwriting, 49 lines copied in FineNotes |
| **iPad, real notes** | 20 | **19.3** | 47.6 | the developer's own lecture-note lines |
| iPad, copy sheet, adapted to the writer | 49 | 3.4 | 15.1 | after tuning on the writer's other lines (cross-validated) |

**What to expect on an iPad.** The public test sets are pen tablets and
whiteboards. On FineNotes ink, the only measurement so far is one writer
(the developer): about 14–19 % of characters wrong. Most errors are digits
and symbols, and words a pen-tablet writer would form differently. After
adapting to that writer, 3.4 %. FineNotes cannot adapt the model on the
device yet; the adapted row shows what that would bring.

## Training data

Every source is cleared for an Apache 2.0 model (see `NOTICE`). No data
under a non-commercial licence was used, and checkpoints were chosen on
held-out UNIPEN writers only.

| Source | Lines per epoch (share) | What it is |
|---|---|---|
| BRUSH | 12,300 (8 %) | 162 writers, short English sentences |
| IAM-OnDB | 12,300 (8 %) | 173 writers, 7,545 whiteboard lines |
| UNIPEN | 12,300 (8 %) | 58,461 words and lines |
| UNIPEN, joined words and characters | 12,300 (8 %) | words joined into lines; character lines |
| PluColl | 12,300 (8 %) | 32 writers' words, single and joined |
| UJIpenchars2 | 6,150 (4 %) | 60 writers' characters, spliced into lines |
| Synthetic lines, BRUSH ink | 58,500 (37 %) | real ink cut into letters and words and spliced into new text: sentences, dates, maths, numbers, notes; some in derived writing styles |
| Synthetic lines, IAM-OnDB ink | 21,500 (14 %) | the same from IAM-OnDB writers |
| Synthetic restyled lines | 9,200 (6 %) | real BRUSH lines redrawn with changed proportions |

Text for synthetic lines and the language model: Common Voice (CC0), arXiv
titles and abstracts (CC0), Faker (MIT), the DeepMind Mathematics Dataset
generator (Apache 2.0), GSM8K (MIT), FineNotes templates and the developer's
own typed notes.

## Training

48 epochs of 157,000 lines, Adam (peak learning rate 7 × 10⁻⁴, 8 % warm-up),
CTC loss, augmentation by rotation, shear, scale and pen speed. The best
epoch on held-out UNIPEN writers was kept. About 7.3 hours on a laptop GPU
(RTX 1000 Ada).

The same recipe without the synthetic lines of 2026-10-09 (spliced joins,
blended letters, the new text) was trained as a control. Against it, this
model is 0.7–1.4 points better on UNIPEN and DeepWriting and 1.2 points
better on the writer-adapted copy sheet, and 0.4 points worse on ImadocSen.

## Known limitations

- **Digits.** Common confusions: 1 and 7, 6 and b, 1 and l.
- **New FineNotes writers are barely measured.** The iPad rows are one
  writer. Feedback and opt-in samples from other writers are what the next
  version needs most.
- **Writing styles far from the training writers** (very cursive, very
  small, left-handed slant) are likely to read worse; this is not
  measured.
- **Synthetic lines** make up about half of training. They are spliced from
  real ink but can carry artefacts, such as joins between letters that a
  person would not write.

## Version history

- **1.0** (this file): first public model.

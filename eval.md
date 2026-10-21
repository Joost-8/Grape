# Evaluation of `grape-2.0.bin` (2026-10-21)

Measured with the file itself through FineNotes' own TypeScript reader: int8 network, pruned LM (min count 32), alpha 0.3, beta 0.0, beam 16. CER in %, greedy / with the language model. Grape 1.0 measured the same way, for comparison.

| Test set | Lines | Grape 1.0 | Grape 2.0 | What it is |
|---|---|---|---|---|
| U | 300 | 3.2 / 2.5 | 3.0 / 2.4 | UNIPEN, held-out writers (US tablets, 1990s) |
| D | 300 | 5.1 / 4.5 | 5.5 / 5.5 | DeepWriting, held-out writers (IAM-OnDB and ETH tablet writers) |
| I | 300 | 4.2 / 3.2 | 3.0 / 1.8 | ImadocSen, French tablet-PC writers (no training source shares them) |
| IM | 300 | 2.2 / 1.5 | 2.6 / 1.6 | IAM-OnDB, held-out writers (whiteboard) |
| A1 | 300 | 2.4 / 2.6 | 1.6 / 2.0 | BRUSH, held-out writers |
| NS3 | 300 | 2.4 / 1.2 | 1.9 / 1.0 | held-out BRUSH writers on held-out note text (dates, maths, numbers) |
| C | 49 | 19.8 / 14.0 | 14.4 / 9.7 | the developer's copy sheet: 49 lines copied in FineNotes on an iPad (exact labels) |
| N39 | 39 | 26.3 / 21.2 | 18.7 / 13.4 | the developer's own lecture-note lines (labels from transcripts) |
| CS3 | 10 | 29.8 / 27.2 | 25.1 / 24.6 | the developer's maths sheet: formulas on one line |
| O | 148 | 32.9 / 32.0 | 29.9 / 28.1 | a second FineNotes writer (labels from transcripts) |

The public sets are 300-line samples. N39's lines were chosen by a reference model's match score, which favours lines that are already read reasonably, so its absolute numbers are optimistic; the comparison between models is the same. CS3 is ten lines and moves by several points between training runs.

After tuning on the copy sheet C alone (300 steps, FineNotes' own tuner), Grape 2.0 reads, with the language model: N39 6.9 (from 13.4), CS3 16.1 (from 24.6), O 27.6 (from 28.1), U 3.8 (from 2.4). Tuning fits the writer it is tuned on.

Speed (laptop, features and network, one line): Grape 1.0 34 ms, Grape 2.0 24 ms.

# Evaluation of `finenotes-hwr-1.0.bin` (2026-10-11)

Measured with the file itself: int8 network, pruned LM (min count 32), alpha 0.3, beta 0.0, beam 16 (`card_eval.py`). CER and WER in %.

| Test set | Lines | Greedy CER | CER | WER | What it is |
|---|---|---|---|---|---|
| U | 600 | 4.1 | 3.4 | 12.7 | UNIPEN, held-out writers (US tablets, 1990s) |
| D | 600 | 5.4 | 4.6 | 18.5 | DeepWriting, held-out writers (IAM-OnDB and ETH tablet writers) |
| I | 600 | 3.9 | 2.9 | 7.3 | ImadocSen, French tablet-PC writers (no training source shares them) |
| IM | 600 | 2.4 | 1.7 | 6.8 | IAM-OnDB, held-out writers (whiteboard) |
| A1 | 600 | 2.1 | 2.4 | 14.0 | BRUSH, held-out writers |
| J | 600 | 1.6 | 0.8 | 3.2 | UJIpenchars2, held-out writers (Spanish, characters) |
| NS3 | 600 | 2.3 | 1.2 | 5.7 | held-out BRUSH writers on held-out note text (dates, maths, numbers) |
| C | 49 | 19.6 | 14.2 | 44.7 | the developer's copy sheet: 49 lines copied in FineNotes on an iPad (exact labels) |
| A2 | 20 | 26.4 | 19.3 | 47.6 | the developer's own lecture-note lines, 20 lines (labels from transcripts) |
| C, adapted to the writer | 49 | 4.7 | 3.4 | 15.1 | the copy sheet after 300 steps of tuning on the writer's other lines (5-fold CV; the plugin cannot do this yet) |

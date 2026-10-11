# The model file format (version 1)

One binary file holds the network, the character language model and the
settings a reader needs. Everything is little-endian, and every block starts
on a 4-byte boundary. [`reader.py`](reader.py) is a complete reader.

```
"FNHW"            4 bytes, magic
u32 version       1
u32 meta bytes
META              UTF-8 JSON, padded with spaces to the meta byte count
sections          until the end of the file
```

Each section is a 4-byte tag, a `u32` payload byte count, and the payload,
padded with zeros to a 4-byte boundary.

## META

The JSON header names the model and its licence, and says how to run it:

- `alphabet`: the characters the network writes. Output class 0 is the CTC
  blank; class `k` is `alphabet[k - 1]`.
- `net`: `inputs`, `hidden`, `layers`, `classes`, `standardize` (whether the
  network section starts with per-channel input mean and standard deviation),
  `quant` (`int8-row`).
- `features`: how ink becomes the network's input (`version`, `norm`,
  `pointsPerUnit`, `heightsPerSecond`, `coreUnits`).
- `decoder`: beam search settings (`alpha` language model weight, `beta`
  word bonus, `beam`, `topk`, `prune`).
- `lm`: present when the file has a language model (`order`, `chars`, `V`,
  `nodes`, `preds`, `minCount`).

## `NET1`: the network

A stack of bidirectional LSTM layers and a linear output layer, with weights
stored as int8 per output row: a weight is `q × scale` of its row.

```
f32 input_mean[I], input_std[I]                  (when meta.net.standardize)
per layer, per direction (forward, then backward):
  f32 wih_scale[4H], whh_scale[4H], bias[4H]     (bias = bias_ih + bias_hh)
  i8  wih[4H × In]   row-major; In = I for layer 0, else 2H
  i8  whh[4H × H]
f32 out_scale[C], out_bias[C]
i8  out[C × 2H]
```

`I` is `meta.net.inputs`, `H` is `meta.net.hidden`, `C` is
`meta.net.classes`. The gate order in every `4H` block is PyTorch's: input,
forget, cell, output.

## `LM01`: the character language model

A character n-gram model stored as a tree of contexts.

```
u32 nodes, u32 preds, u32 count bytes
u8  node_char[nodes], node_children[nodes], node_preds[nodes]
u8  pred_char[preds]
LEB128 counts[preds]                             (count bytes in total)
```

- Nodes are contexts in breadth-first order. Node 0 is the empty context.
- A node's children reach one character further into the past. They are
  contiguous and sorted by character code.
- A node's predictions are contiguous and sorted by character code. A node's
  total count and type count are the sum and the number of its predictions.
- Character codes index `meta.lm.chars`; code 0 is the start padding.

The probabilities are Witten-Bell interpolated over the context tree.
FineNotes' reader (`src/recognition/hwr/char-lm.ts`) is the reference for
how the language model is scored during decoding.

# MidiMaker — Design Document

> **Version:** 0.1.0  
> **Date:** 2024  
> **Status:** Foundational Architecture

---

## 1. Executive Summary

MidiMaker is a local-first Windows 10 desktop application for symbolic music generation. It lets musicians compose, continue, inpaint, and personalise MIDI music through a PySide6 GUI backed by a small transformer model (GPT-2-small adapted to MIDI tokens), PEFT LoRA style packs, and an evolutionary training pipeline — all running entirely offline.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PySide6 GUI (app/)                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │  Compose │  │  Library │  │  Train   │  │  Rank    │  │  Settings   │  │
│  │  Screen  │  │  Screen  │  │  Screen  │  │  Screen  │  │  Screen     │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬───────┘  │
└───────┼─────────────┼─────────────┼──────────────┼──────────────┼──────────┘
        │             │             │              │              │
        ▼             ▼             ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Application Core (core/)                             │
│   MidiMakerProject · MidiPiece · StylePack · utils · midi_representation   │
└────────────┬─────────────────────────────────────────────────────────────┬──┘
             │                                                             │
     ┌───────▼────────┐                                         ┌─────────▼──────┐
     │  Generation    │                                         │   Training     │
     │  Subsystem     │                                         │   Subsystem    │
     │  (generation/) │                                         │  (training/)   │
     │                │                                         │                │
     │ · Inpainting   │                                         │ · Preprocessor │
     │ · Continuation │                                         │ · Evaluator    │
     │ · Motif        │                                         │ · StylePack    │
     │ · Bassline     │                                         │ · Evolutionary │
     └───────┬────────┘                                         └────────────────┘
             │
     ┌───────▼────────┐         ┌─────────────────────────────┐
     │  Adapters      │         │   Ranking Subsystem         │
     │  (adapters/)   │         │   (ranking/)                │
     │                │         │                             │
     │ · Tokenizer    │         │ · PreferenceStore           │
     │ · ModelWrapper │         │ · PreferenceReranker        │
     │ · StyleAdapter │         └─────────────────────────────┘
     └────────────────┘
```

### 2.1 Execution Environments

| Context | Where it runs | Notes |
|---------|--------------|-------|
| GUI thread | PySide6 main thread | UI updates only |
| Generation | QThreadPool worker | emits progress signals |
| Training | QThreadPool worker (long-running) | can be cancelled |
| MIDI I/O | synchronous, fast | inline in GUI thread acceptable for <1 MB files |

---

## 3. Subsystem Designs

### 3.1 Generation Engine

**Primary model:** GPT-2-small (124 M parameters) fine-tuned on a large MIDI token corpus (e.g., GigaMIDI-derived token sequences).  The base checkpoint is stored in `assets/base_model/`. LoRA adapters overlay style packs at inference time.

**Tokenisation scheme (selectable):**  
- Default: **REMI** (Relative Event-based MIDI representation) via MIDITok ≥3.0.  
- Alternative: **TSD** (Token Shift Duration) or **Structured** — configurable per style pack.

REMI event vocabulary covers: `Bar`, `Position`, `Pitch`, `Velocity`, `Duration`, and programme-change tokens.  Vocabulary size ≈4096 (special tokens + pitch 0-127 × 4 velocity bins × 32 duration values + structural tokens).

**Inference flow:**

```
MidiPiece  ──tokenize──►  token ids  ──prepend context──►  model.generate()
                                                               │
                              ◄─────── output token ids ───────┘
                              │
                        detokenize ──►  MidiPiece  ──anti-copy check──►  GenerationResult
```

**Sampling:**  
- Temperature ∈ [0.6, 1.2] (user-adjustable).  
- Top-p (nucleus) sampling with p = 0.92.  
- Repetition penalty = 1.2 applied to token ids already seen in context.

### 3.2 Tokenisation

MIDITok `REMITokenizer` wraps `MidiTokenizerWrapper` (adapters/midi_tokenizer.py).  The wrapper handles:
- Loading/saving the vocabulary to `assets/vocab/remi_vocab.json`
- Padding and attention masks for batched generation
- Special infilling tokens `<MASK>`, `<SEP>`, `<BOS>`, `<EOS>` used during inpainting

### 3.3 Style Adaptation (MIDI LoRA)

**What is learned:**  
A LoRA adapter (rank r=8, alpha=32) on the Q, V projection matrices of every attention layer in GPT-2-small.  Total trainable parameters ≈ 300 K — roughly 1.2 MB at float32.  The file also bundles:
- Token-level bias vectors for groove/velocity adjustment (≈32 KB)
- A small learned linear head for bar-level energy prediction (≈4 KB)
- Metadata JSON (creator, date, source MIDI count, evaluation scores)

**Total style pack size:** 2–4 MB (well within the 50 MB target).

**How loading works:**
1. Base model loaded once at startup (or lazy on first use).
2. On style pack activation, `peft.set_peft_model_state_dict()` injects adapter weights.
3. Switching styles swaps only LoRA weight tensors — no full model reload.
4. The TokenBias and EnergyHead are applied as a post-processing pass over logits.

**Training data requirement:**  ≥ 50 MIDI files (≈ 200 bars each) is sufficient for a meaningful style pack.  Practical upper bound is limited only by disk and time.

### 3.4 Motif Discovery

`MotifAnalyzer` uses a sliding window of 2–8 notes over each melody track:
1. Each window is embedded to a fixed-length vector using the base model's encoder hidden states (the average of last-layer token embeddings for that window's tokens).
2. Cosine similarity matrix between all windows is computed.
3. Agglomerative clustering (scikit-learn) groups similar windows into motif families.
4. Transposition-invariance: pitch-class content is shifted to root=0 before embedding.
5. Rhythmic-variation tolerance: duration ratios rather than absolute values are encoded.

Each `Motif` records the canonical note list, all occurrence bar/beat positions, a plain-language description (e.g., "rising 3-note figure, dotted rhythm"), and a confidence score.

### 3.5 Bassline Generation

Strategy:
1. Parse chord progression from input (user-annotated or detected via music21 harmonic analysis).
2. Choose style (`simple`, `driving`, `syncopated`, `melodic`, `legato`, `staccato`).
3. Template-based skeleton: root on beat 1, optional 5th/octave approach on beat 3 (for simple/driving).
4. Infill with model conditioned on chord tokens and style pack.
5. Anti-dissonance post-processing: any non-chord note longer than one 16th note is shifted to nearest chord tone unless it forms a recognised passing note (stepwise motion).

### 3.6 Preference Learning

The preference learning pipeline is intentionally lightweight and transparent:

1. User assigns `like/dislike/favourite/discard` ranks to generated results.
2. `PreferenceStore` saves ranked pairs to `preference_log.json` in the project folder.
3. `PreferenceReranker` trains a 2-layer MLP (64→32→1) using (feature_vector, label) pairs:
   - Feature = 12-dim: pitch_class_hist similarity, note_density, groove_score, range, polyphony, key_consistency × 2 + copying_risk + 4 user-feedback accumulators.
   - Label = 1 (liked/favourite) or 0 (disliked/discarded).
4. The MLP is trained with Adam (lr=1e-3) for up to 50 epochs whenever the store grows.
5. At inference time, candidates are re-sorted by predicted preference score before display.

---

## 4. Evolutionary Refinement Loop

**Goal:** Find the LoRA hyperparameter configuration (rank, alpha, dropout, learning-rate schedule, data augmentation policy) and generation settings (temperature, top-p, repetition penalty) that maximise the composite evaluation score on a held-out set of the user's own MIDI files.

```
┌─────────────────────────────────────────────────────────────────┐
│  Population (N=20 individuals)                                  │
│  Each individual = {rank, alpha, dropout, lr, augment_flags,    │
│                     temperature, top_p, rep_penalty}            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  Evaluate each individual       │
          │  · Train adapter for K epochs   │
          │  · Generate N_eval samples      │
          │  · Compute composite score      │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  Selection (top-50% tournament) │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  Crossover (uniform)            │
          │  Mutation (Gaussian noise,      │
          │            p_mut=0.15)          │
          └────────────────┬────────────────┘
                           │
                     next generation
```

**Convergence criterion:** Improvement < 0.005 over 3 consecutive generations, or max_generations reached (default 20).

**Wall-clock estimate:** With a GPU (RTX 3060 or better), one full evolution run (20 individuals × 10 eval epochs × 20 generations) takes approximately 45–90 minutes. CPU-only: 6–12 hours (not recommended).

---

## 5. Data Flow Diagrams

### 5.1 Training a Style Pack

```
User MIDI files (N ≥ 50)
        │
        ▼
MidiPreprocessor.parse_midi()  ──►  MidiPiece[]
        │
        ▼
split_tracks() ──► detect_key() ──► detect_tempo_map()
        │
        ▼
MidiTokenizerWrapper.tokenize()  ──►  token sequences []
        │
        ▼
HuggingFace Dataset (in-memory or HDF5 cache)
        │
        ▼
StyleAdapter.train_adapter()  [PEFT LoRA fine-tune, N epochs]
        │
        ▼
MusicEvaluator.overall_score()  [evaluate on held-out split]
        │
        ▼
EvolutionaryRefiner.evolve()  [optional hyperparameter search]
        │
        ▼
StylePackCreator.save()  ──►  <name>.midilora  file on disk
```

### 5.2 Generating with a Style Pack

```
Loaded MidiPiece (context)  +  StylePack
        │
        ▼
MidiTokenizerWrapper.tokenize(context)  ──►  context_tokens
        │
        ▼
SymbolicMusicModel.load_style_adapter(style_pack)
        │
        ▼
SymbolicMusicModel.generate(context_tokens, …)  ──►  output_tokens
        │
        ▼
MidiTokenizerWrapper.detokenize(output_tokens)  ──►  raw MidiPiece
        │
        ▼
MusicEvaluator: score, anti_copying_penalty, copying_risk_score
        │
        ▼
PreferenceReranker.rerank([candidates])  ──►  ranked GenerationResult[]
        │
        ▼
GUI displays ranked candidates
```

---

## 6. Evaluation Function Details

The composite evaluation score weights six sub-scores:

| Component | Weight | Description |
|-----------|--------|-------------|
| pitch_class_distribution_similarity | 0.20 | KL divergence between generated and style-pack pitch histogram |
| groove_similarity | 0.20 | Cross-correlation of 16th-note onset density profiles |
| key_consistency_score | 0.15 | Fraction of notes belonging to detected key |
| harmonic_compatibility | 0.15 | Fraction of notes compatible with chord progression |
| phrase_end_plausibility | 0.10 | Does the phrase end on a stable degree (1, 3, 5)? |
| motif_shape_similarity | 0.10 | Cosine similarity to nearest motif in context |
| anti_copying_penalty | 0.10 | Penalises n-gram (n=4 note tokens) overlap with training corpus |

The `anti_copying_penalty` subtracts from the score; all other components add to it.  The `copying_risk_score` (0–1) is reported separately and displayed in the GUI as a plain-language label ("Novel" / "Derivative" / "Flagged").

---

## 7. Anti-Copying / Novelty Guards

MidiMaker implements a two-layer novelty guard:

**Layer 1 — Token n-gram filter:**  
During inference, the model's logit distribution is biased away from any continuation that would exactly replicate a 4+ note sequence found in the training corpus (stored as a Bloom filter for memory efficiency).

**Layer 2 — Post-generation evaluation:**  
`MusicEvaluator.copying_risk_score()` computes the fraction of 6-note windows in the generated piece that appear verbatim (up to octave transposition) in the training set. If the score exceeds 0.25, the result is flagged in the UI and the overall score is penalised by 0.3.

These guards do **not** claim copyright-level originality; they are practical heuristics to keep generated music fresh and to reduce risk of inadvertent reproduction.

---

## 8. File & Persistence Formats

| Artefact | Extension | Format | Typical size |
|----------|-----------|--------|--------------|
| Project | `.midimaker` | JSON (gzip-compressed) | 10–100 KB |
| Style Pack | `.midilora` | JSON header + PEFT safetensors | 2–4 MB |
| Exported MIDI | `.mid` | Standard MIDI File type 1 | 5–50 KB |
| Token cache | `.h5` | HDF5 dataset | 10–200 MB |
| Preference log | `preference_log.json` | JSON array | < 1 MB |

---

## 9. GUI Screen Map

```
┌─────────────────────────────────────────────────────────────────┐
│  MidiMaker                                            [─][□][×] │
│  ┌────────┬──────────┬─────────┬────────┬────────────────────┐  │
│  │Compose │ Library  │ Train   │  Rank  │    Settings        │  │
│  └────┬───┴──────────┴─────────┴────────┴────────────────────┘  │
│       │                                                          │
│  ┌────▼────────────────────────────────────────────────────┐    │
│  │  Piano Roll  (read-only preview)                        │    │
│  │  ────────────────────────────────────────────────────── │    │
│  │  [Import MIDI]  [Style: <pack>▼]  [Generate ▼]         │    │
│  │                                                         │    │
│  │  Candidates:  ○1  ○2  ○3  ○4   [Play] [Export] [Rank] │    │
│  │  Score: 0.87  Novelty: Novel   Copying Risk: Low        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Security & Privacy

- **Fully offline:** No network calls at runtime.
- **No telemetry:** No analytics, crash reporting, or usage tracking.
- **Local model weights:** The base model checkpoint ships with or is downloaded once to `assets/base_model/` and verified by SHA-256.
- **User MIDI files:** Processed only in memory and written back to project folder; never transmitted.

---

## 11. Known Limitations & Future Work

- The base model (~124 M params) is modest; larger models (MusicGen-style) can be substituted by implementing a new `BaseGenerator` subclass.
- Expressive performance (timing humanisation, continuous controller curves) is out of scope for v0.1; only symbolic pitch/duration/velocity are generated.
- Multi-track coherent generation (e.g., melody + bass + chords simultaneously) is planned for v0.3 via cross-track conditioning tokens.
- Real-time playback requires a separate MIDI synthesizer (e.g., FluidSynth + soundfont); the app launches the user's default system MIDI device.

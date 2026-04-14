# MidiMaker — Local Symbolic Music Workstation

MidiMaker is a desktop application for Windows that helps you compose original music using AI.
It learns your personal style from MIDI files you already own, then helps you fill gaps in the
piano roll, continue unfinished clips, discover recurring motifs, and generate basslines — all
running 100 % locally on your own computer with no internet connection required after setup.

---

## Table of Contents

1. [What MidiMaker Does](#what-midimaker-does)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [First Run](#first-run)
5. [Creating a Project](#creating-a-project)
6. [Importing MIDI Files](#importing-midi-files)
7. [Training a Style Pack](#training-a-style-pack)
8. [Filling a Gap (Inpainting)](#filling-a-gap-inpainting)
9. [Continuing a MIDI Clip](#continuing-a-midi-clip)
10. [Finding Motifs](#finding-motifs)
11. [Generating Motifs from Chords](#generating-motifs-from-chords)
12. [Creating Basslines](#creating-basslines)
13. [How Ranking Works](#how-ranking-works)
14. [Where Files Are Stored](#where-files-are-stored)
15. [Backing Up Projects](#backing-up-projects)
16. [Troubleshooting](#troubleshooting)
17. [License](#license)

---

## What MidiMaker Does

MidiMaker analyses MIDI files from your music library and builds a compact *Style Pack* — a tiny
AI model adapter trained on your collection.  Once trained, you can use the style pack to:

| Feature | Plain-English description |
|---|---|
| **Inpainting** | You select a section of the piano roll, delete it, and MidiMaker writes new notes that fit the music on both sides. |
| **Continuation** | Give MidiMaker the first half of a clip and it writes the next 4–16 bars in the same style. |
| **Motif finder** | MidiMaker spots repeated melodic ideas in your piece and highlights them. |
| **Motif generator** | Pick a chord progression and MidiMaker invents short melodic phrases that fit. |
| **Bassline generator** | Provide a chord chart and choose a groove style; MidiMaker writes a matching bass part. |
| **Ranking** | You thumbs-up or thumbs-down every result, and MidiMaker learns your preferences over time. |

Everything runs on your machine — no subscription, no cloud upload, no monthly fee.

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Operating system | Windows 10 (64-bit) | Windows 11 (64-bit) |
| Python | 3.10 | 3.11 or 3.12 |
| RAM | 8 GB | 16 GB |
| Disk space | 4 GB free | 8 GB free |
| GPU | None (CPU mode) | NVIDIA GPU with 6 GB VRAM |
| CUDA (NVIDIA only) | — | CUDA 11.8 or 12.x |

> **Note:** An NVIDIA GPU speeds up generation dramatically (a few seconds vs. a few minutes).
> The app works fine on CPU — it just takes longer.

---

## Installation

> You only need to do this once.

### Step 1 — Install Python

1. Go to <https://www.python.org/downloads/> and download the latest **Python 3.11** or **3.12** installer.
2. Run the installer.
3. **Important:** tick the checkbox **"Add Python to PATH"** before clicking Install.
4. When the installer finishes, click **"Disable PATH length limit"** if prompted, then close.

To verify, open a Command Prompt (`Win + R`, type `cmd`, press Enter) and run:

```
python --version
```

You should see something like `Python 3.11.9`.

### Step 2 — Download MidiMaker

Download or clone this repository to a folder on your computer, for example `C:\MidiMaker`.

### Step 3 — Run the Setup Script

1. Open File Explorer and navigate to the MidiMaker folder.
2. Double-click **`setup.bat`**.
3. A Command Prompt window opens and runs seven steps automatically:
   - Checks your Python version
   - Creates an isolated Python environment
   - Installs all dependencies (~548 MB download on first run)
   - Downloads the base AI model from HuggingFace
   - Checks whether a GPU is available
   - Creates the `projects/`, `models/`, and `logs/` folders
   - Runs a quick smoke test
4. When you see **"Setup complete!"**, close the window.

If any step fails, read the error message — it usually tells you exactly what to fix.

---

## First Run

Double-click **`run.bat`** in the MidiMaker folder.

The application window opens showing the **Home** screen.  The first time you run it, no style
packs are listed — that is normal.  Create a project and train your first style pack (see below).

---

## Creating a Project

A *project* keeps all your work for a single song or session together — imported MIDI files,
style packs, generated clips, and your ratings.

1. Click **New Project** on the Home screen.
2. Type a name (e.g. `Jazz Ballad`).
3. Click **Create**.

Your project is saved as a single `.midimaker` file inside the `projects/` folder.

---

## Importing MIDI Files

You can import as many MIDI files as you like into a project.  These are the files MidiMaker
will learn from.

1. Open your project and go to the **Piano Roll** screen.
2. Click **Import MIDI…** in the toolbar.
3. Browse to a `.mid` or `.midi` file and click **Open**.

MidiMaker reads the file, detects the key, tempo, and track roles (melody, bass, chords, drums),
and shows a summary.  Repeat for as many files as you want.

> **Tip:** Import at least 5–10 MIDI files in a similar style for the best training results.

---

## Training a Style Pack

A *Style Pack* is the AI's memory of your style.  Training teaches it the rhythms, note choices,
and harmonic patterns found in your imported MIDI files.

1. Go to the **Training** screen.
2. Select the MIDI files you want to train on (or click **Select All**).
3. Give the Style Pack a name (e.g. `My Jazz Style`).
4. Click **Train**.

Training takes a few minutes on a GPU, or 15–30 minutes on CPU.  A progress bar shows the
current epoch.  When it finishes, the Style Pack appears in the list and is ready to use.

**What is actually happening?**
The base model is a small transformer pre-trained on general MIDI patterns.  Training applies a
technique called *LoRA* (Low-Rank Adaptation) to steer it toward your specific style without
changing the base model — so each Style Pack is tiny (a few MB) and multiple packs can coexist.

---

## Filling a Gap (Inpainting)

*Inpainting* fills in a section of the piano roll that you have deleted or left blank.

1. Open your project and go to the **Piano Roll** screen.
2. Load a MIDI file using **Import MIDI…**.
3. Select a range of bars you want to replace (click and drag on the bar ruler).
4. Press **Delete** to clear the selection.
5. Open the **Inpainting** panel on the right.
6. Choose a **Style Pack** and set **Candidates** (how many options to generate, 1–8).
7. Click **Generate**.

MidiMaker produces several candidate fills.  Each one shows a star rating and a *Copying Risk*
indicator (how similar it is to your training data — lower is more original).  Click any
candidate to preview it in the piano roll, then click **Accept** to keep it.

---

## Continuing a MIDI Clip

*Continuation* adds new bars after the end of an existing clip.

1. Go to the **Continuation** screen.
2. Load or select a MIDI clip as the prompt.
3. Set **Length** (how many bars to add, 4–32).
4. Choose a **Style Pack**.
5. Click **Generate**.

---

## Finding Motifs

A *motif* is a short musical idea that repeats throughout a piece — think of the four-note
opening of Beethoven's Fifth Symphony.  MidiMaker finds these automatically.

1. Go to the **Motif** screen.
2. Load a MIDI file.
3. Click **Analyse**.

MidiMaker highlights every occurrence of each motif in the piano roll using a different colour.
You can click a motif to see all the bars where it appears.

---

## Generating Motifs from Chords

Once you have found (or typed in) a chord progression, MidiMaker can invent melodic motifs that
fit the harmony.

1. On the **Motif** screen, click **Chord Input** at the top.
2. Type your chord progression, one chord per bar (e.g. `C  G  Am  F`).
3. Choose a **Style Pack** and a **Mode**:
   - *Close* — stays close to the original motif
   - *Same Feel* — keeps the mood but explores more
   - *Exploratory* — more adventurous variations
4. Click **Generate Motif**.

---

## Creating Basslines

1. Go to the **Bassline** screen.
2. Enter your chord progression using the Chord Input widget.
3. Choose a **Bass Style**:
   - *Simple* — one root note per bar
   - *Driving* — eighth-note pulse on the root
   - *Syncopated* — off-beat rhythms
   - *Melodic* — scalar runs between chord roots
   - *Legato* — smooth, connected notes
   - *Staccato* — short, punchy notes
4. Click **Generate**.

---

## How Ranking Works

Every time MidiMaker generates something, you can rate it:

| Button | Meaning |
|---|---|
| ♥ Favourite | Excellent — use this as a model for future generations |
| 👍 Like | Good — keep it |
| 👎 Dislike | Not right — avoid this direction |
| 🗑 Discard | Delete completely |

Ratings are stored in `preference_log.json` inside your project folder.  Over time, MidiMaker
uses your ratings to re-rank candidates, showing you better results sooner.

---

## Where Files Are Stored

```
MidiMaker/
├── projects/
│   └── My_Jazz_Ballad/
│       ├── My_Jazz_Ballad.midimaker   ← project file (gzip JSON)
│       └── preference_log.json        ← your ratings
├── models/
│   └── base_model/                    ← downloaded GPT-2 base (one-time)
├── logs/
│   └── midimaker_2024-01-01.log       ← diagnostic log
└── assets/
    └── vocab/                         ← MIDI tokeniser vocabulary
```

Style Pack adapter weights are stored inside the project folder when you train them.

---

## Backing Up Projects

To back up a project:

1. Copy the entire `projects/My_Project_Name/` folder to an external drive or cloud storage.
2. That folder contains everything — the project file, your ratings, and the style pack adapters.

To restore:

1. Copy the folder back into `projects/`.
2. Open MidiMaker and click **Open Project**, then browse to the `.midimaker` file.

---

## Troubleshooting

### "Python not found" during setup
Make sure you ticked **"Add Python to PATH"** during installation.  If not, uninstall Python and
reinstall it with that option enabled.

### "pip install failed" — network error
Check your internet connection.  If you are behind a corporate proxy, ask your IT department for
the proxy address and set it:

```
set HTTPS_PROXY=http://proxy.example.com:8080
setup.bat
```

### "Model download failed"
Run `setup.bat` again.  If it still fails, download manually:
1. Visit <https://huggingface.co/gpt2> and download all files.
2. Place them in `models/base_model/`.
3. Run `python scripts/download_assets.py --offline` to validate.

### Application opens but generation is very slow
You are running in CPU mode.  Check the **Settings** screen — it shows the detected device.
To use your NVIDIA GPU, install CUDA: <https://developer.nvidia.com/cuda-downloads>

### "CUDA out of memory" error
Reduce the number of candidates (try 1 or 2) and reduce **Max New Tokens** in Settings.

### Log files
Detailed logs are written to `logs/midimaker_YYYY-MM-DD.log`.  Include the relevant log file
when reporting a bug.

---

## License

MidiMaker is released under the **MIT License**.

```
MIT License

Copyright (c) 2024 MidiMaker Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Third-party dependency licences are listed in [THIRD_PARTY_AUDIT.md](THIRD_PARTY_AUDIT.md).

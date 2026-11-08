# Data notes: ds005028

Source: Chandravadia et al. (2024), OpenNeuro ds005028,
doi:10.18112/openneuro.ds005028.v1.0.0
Paper: Chandravadia et al. (2022), medRxiv 2022.06.24.22276882

## Files
- Each recording exists as .edf and .dat. The .edf files can't be used: header claims
  1 s duration, channels unnamed (Signal_1-32), no events.
- We use the BCI2000 .dat files, read with BCI2kReader.
- Full per-file survey: docs/dataset_survey.csv (from scripts/00c_survey.py)

## Recording
- Sampling rate: 256 Hz
- 32 channels: Fz FC1 FCz FC2 FC4 FC6 C4 C6 CP4 CP6 FC3 FC5 C3 C5 CP3 CP5 CP1 P1
  Cz CPz Pz POz CP2 P2 PO7 PO3 O1 Oz Iz O2 PO4 PO8
- No P3/P4 in montage. Using: Fz Cz CPz Pz POz P1 P2 PO7 PO8 Oz
- Signal units: microvolts (Pz std ~10-30 for most runs); converted to volts for MNE

## Sessions and paradigms
- FlashingParadigm codes: 0 = rc (row/column), 1 = cb (checkerboard), 2 = comb
- ses-train: copy spelling (InterpretMode 2), labeled targets (StimulusType)
    - rc:   1200 flashes/run, codes 1-12,  target fraction 0.167
    - cb:   1800 flashes/run, codes 1-180, target fraction 0.111
    - comb: 1100 flashes/run, codes 1-110, target fraction 0.182
- ses-test: free spelling (InterpretMode 1), TextToSpell empty, StimulusType all 0
  (no labels). Participants typed their own sentences (see TextResult).
  Mixed paradigms, variable-length sequences (online LM + dynamic stopping).
  -> NOT USED (no ground truth).

## rc speller (what we simulate)
- Grid: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'sp']
- Flash duration 62.5 ms; gap 62.5 ms; flash onset every 125 ms (32 samples)
- 12 flashes per sequence (6 rows + 6 columns); 10 sequences per character
- Pre-character pause 2000 ms; post-character pause 1500 ms
- Time per character (baseline): 10*12*0.125 + 2.0 + 1.5 = 18.5 s
  (recorded run is ~202 s for 10 chars; extra time is start/end padding)
- Code convention: B. StimulusCode 1-6 = ROWS (top to bottom),
  7-12 = COLUMNS (left to right). Verified: decoded text matches TextToSpell
  for sub-01 ('THE QUICK ', 'BROWN FOX '). Space key shows as 'sp'.
  Grid index = (row-1)*6 + (col-1)
- Preprocessing: 0.5-20 Hz band-pass, 60 Hz notch, average reference (32 ch),
  epochs -0.1 to 0.8 s, baseline -0.1 to 0, downsampled to 64 Hz
  (>= 3x low-pass, avoids aliasing), features = 0-0.8 s x 10 channels (~520)

## Phrases (same for all 11 subjects)
- rc run 1 of each subject: 'THE QUICK ' (10 characters incl. trailing space)
- rc run 2 of each subject: 'BROWN FOX ' (10 characters incl. trailing space)
- Note: from a well-known pangram -> LLM predictions may be optimistic.
  Plan: also test held-out everyday phrases with simulated selections.

## rc run numbers per subject (ses-train)
| sub | THE QUICK | BROWN FOX |
|-----|-----------|-----------|
| 01  | run 2     | run 5     |
| 02  | run 3     | run 6     |
| 03  | run 2     | run 5     |
| 04  | run 2     | run 5     |
| 05  | run 2     | run 5     |
| 06  | run 3     | run 6     |
| 07  | run 2     | run 5     |
| 08  | run 1     | run 4     |
| 09  | run 1     | run 4     |
| 10  | run 3     | run 6     |
| 11  | run 2     | run 5     |

## Events (BCI2000 states, one value per sample)
- StimulusCode: 0 = no flash; 1-12 = which row/column is flashing (rc)
- Flash onset = sample where StimulusCode goes from 0 to non-zero
- StimulusType: 1 = flash contained the target; 0 = not
- New character = gap > 1 s between flash onsets (120 flashes per character)

## Classifier results

- ERP (Part 8): P300 difference at Pz (250-500 ms) small for most participants
  (-0.2 to +1.5 uV), likely reduced by overlapping epochs (125 ms between flashes)
  and average reference. Largest: P10 (1.53), P11 (0.95), P07 (0.88).
- Very noisy: P05 (noise_sd 117.7 uV) and P10 (357.6 uV, likely one bad channel).
  Inclusion decided by the pre-set AUC < 0.6 rule (Part 9).

## Train/test split
- Only ses-train, task-rc (2 runs per subject, 10 characters each).
- 2-fold cross-validation: train THE QUICK -> test BROWN FOX, and the reverse.
- All tuned settings (Bayes threshold, LLM weight) chosen on the training run only.
- ~20 test characters per participant.

## Data quality
- sub-05: train runs very noisy (Pz std: rc 1103 and 549; comb up to 3622 vs
  ~10-30 for others). Kept for now; check in ERP plot (Part 8) and AUC (Part 9).
- Inclusion rule (decide BEFORE seeing ITR results): TODO
  (e.g., exclude participants with test AUC < 0.6, and report it)

## Open questions
- Exact meaning of cb/comb flash groups (not needed for our rc experiment)
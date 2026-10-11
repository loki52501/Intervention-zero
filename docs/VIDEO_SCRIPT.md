# Intervention Zero: 60-second demo video script

Target length 60 s, 1920x1080, calm unhurried voice at about 140-150 wpm.

**Honesty note.** UI footage replays `out/demo_events.jsonl`, a SYNTHETIC stream (SYNTHETIC badge visible). Do not cite numbers in the narration unless they come from a measured run; ablation numbers shown are placeholders until a real run replaces `out/ablation.json`.

## 1. How to produce it

```powershell
.venv\Scripts\python.exe scripts\record_demo.py --check-only
.venv\Scripts\python.exe scripts\record_demo.py --speed 4 --ablation-card --tts
```

Outputs, all under `out/video/`:

- `raw.webm` - UI replay recording (1920x1080)
- `ablation.webm` - 10 s ablation card (only with `--ablation-card`)
- `narration.mp3` - ElevenLabs narration (only with `--tts` and a key, or your own recording)
- `demo.mp4` - raw.webm + narration.mp3, muxed (only when both exist)
- `demo_silent.mp4` - raw.webm transcoded to mp4, no audio
- `cues.txt` - on-screen time of each beat (takeover, fix, gate, promote, reload, drift, clip), computed from the server replay pacing; use it to trim. Times are relative to page load, add 1-2 s.
- `first_frame.png`, `last_frame.png` - sanity screenshots

Manual edit recipe: cut the UI replay into the beats with `-ss/-t` using `cues.txt`, insert `ablation.webm` at 0:46, then concat. Get the ffmpeg binary path with:

```powershell
.venv\Scripts\python.exe -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"
```

Example (substitute that path for `FFMPEG`):

```powershell
FFMPEG -y -ss 0 -t 46 -i out\video\raw.webm -c:v libx264 -pix_fmt yuv420p -r 30 -an out\video\part1.mp4
FFMPEG -y -i out\video\ablation.webm -c:v libx264 -pix_fmt yuv420p -r 30 -an out\video\part2.mp4
FFMPEG -y -ss 46 -t 4 -i out\video\raw.webm -c:v libx264 -pix_fmt yuv420p -r 30 -an out\video\part3.mp4
# list.txt contains: file 'part1.mp4' / file 'part2.mp4' / file 'part3.mp4'
FFMPEG -y -f concat -safe 0 -i out\video\list.txt -c copy out\video\edit.mp4
FFMPEG -y -i out\video\edit.mp4 -i out\video\narration.mp3 -map 0:v -map 1:a -c:v copy -c:a aac -shortest out\video\demo.mp4
```

## 2. Timed shot list

| Time | Shot (what's on screen, DOM element) | Narration | Source event kinds |
|---|---|---|---|
| 0:00-0:08 | Hook, then the floor plan (`#floor`) with robots moving; SYNTHETIC badge visible | "A robot that needs a human every fifth mission isn't autonomous. Here is a simulated hospital floor, and a fleet that keeps getting stuck." | site_loaded, robot_move |
| 0:08-0:22 | Takeover, then `#fix-card`, then `#gate-panel` bars (train / held-out / regression / safety), `#gate-stamp` PROMOTED, reload; `#ticker` lights up | "When an operator takes over, a fixer agent proposes a harness edit. The gate tests it on training incidents, held-out incidents, safety checks and a placebo. It passes, it's promoted, and every robot reloads live." | takeover, fix_proposed, gate_result (C-001), promoted, reload, ticker |
| 0:22-0:32 | Validator refuses S-2 (ticker); gate catches a regression (regression 39/40 in the synthetic stream) | "Unsafe ideas never get that far. The database validator refuses a rule into the human-only zone, S-two, and the gate rejects a fix that breaks an old mission." | validator_refused (C-003), ticker, gate_result (C-004) |
| 0:32-0:46 | `#drift-banner` renovation drift alarm, healing, then the MuJoCo before/after clip in `#clip-overlay` (clips from `out/clips` if present, otherwise [PLACEHOLDER: MuJoCo clip pending]) | "Then the building changes. A renovation pushes takeovers up, and a drift alarm fires. The fleet heals with a new, tested fix, and MuJoCo physics judges the door before and after." | drift_alarm, gate_result (C-005), promoted, clip |
| 0:46-0:56 | Ablation bars with n and 95% CI (`out/video/ablation.webm`); shows the PLACEHOLDER banner while fake_llm is true | "We compare against baseline, placebo and random edits, with sample size and confidence interval on screen." | none (from `out/ablation.json`) |
| 0:56-1:00 | Close card: pitch line and "The world is simulated." | "Intervention Zero turns every takeover into a tested, permanent fix. One honest limit: the world is simulated." | none |

## 3. TTS-ready narration

<!-- TTS-BEGIN -->
A robot that needs a human every fifth mission isn't autonomous. Here is a simulated hospital floor, and a fleet that keeps getting stuck. When an operator takes over, a fixer agent proposes a harness edit. The gate tests it on training incidents, held-out incidents, safety checks and a placebo. It passes, it's promoted, and every robot reloads live. Unsafe ideas never get that far. The database validator refuses a rule into the human-only zone, S-two, and the gate rejects a fix that breaks an old mission. Then the building changes. A renovation pushes takeovers up, and a drift alarm fires. The fleet heals with a new, tested fix, and MuJoCo physics judges the door before and after. We compare against baseline, placebo and random edits, with sample size and confidence interval on screen. Intervention Zero turns every takeover into a tested, permanent fix. One honest limit: the world is simulated.
<!-- TTS-END -->

Verify the word count (must be 120-160) with:

```powershell
.venv\Scripts\python.exe scripts\record_demo.py --check-script
```

Per-beat word budgets (about 2.4 words per second): hook 0:00-0:08 about 24 words; fix and gate 0:08-0:22 about 34; refusal and regression 0:22-0:32 about 24; drift and MuJoCo 0:32-0:46 about 34; ablation 0:46-0:56 about 18; close 0:56-1:00 about 17 (the close runs into the final beat, so trim a clause if it overruns).

## 4. ElevenLabs notes

- Env vars: `ELEVENLABS_API_KEY` (required for `--tts`), `ELEVENLABS_VOICE_ID` (optional; `--voice-id` also works).
- Model `eleven_multilingual_v2`, stability 0.6, similarity 0.75, style 0.
- The script makes exactly one request per run, to protect credits.
- Never commit keys; keep them in `.env` only.

## 5. Record your own voice

Read the TTS block at a calm pace (about 55-60 s) into any recorder and save it as `out/video/narration.mp3`, or convert it:

```powershell
FFMPEG -i voice.m4a -c:a libmp3lame -q:a 2 out/video/narration.mp3
```

Then rerun with `--no-capture` so the existing `raw.webm` is muxed with your audio:

```powershell
.venv\Scripts\python.exe scripts\record_demo.py --no-capture
```

Tips: quiet room, 15 cm from the mic, one breath per beat.

The world is simulated.

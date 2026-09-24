# Minesweeper Benchmark

A text benchmark for whether a model can deduce the next minesweeper move from a correct visible board. The Rust engine is the only implementation of the rules. The Python harness renders that board, calls an OpenAI-compatible chat API, parses the reply, and scores it.

Every model replays the same frozen boards. The hidden mine map stays in the fixture. The opening click is given and is not a reasoning step.

## Layout

| Path | Role |
| --- | --- |
| `engine/` | Boards, no-guess generation, the tiered oracle, and the `msengine` binary |
| `harness/` | The `msbench` uv project (Python 3.13) |
| `viewer/` | A local page for `runs/*.jsonl` |
| `runs/` | Generated boards and JSONL logs (gitignored) |

## Requirements

- A current Rust toolchain
- [uv](https://docs.astral.sh/uv/) with Python 3.13

The engine depends on `ms_toollib` 1.5.16 with its default features, and pins `kstring` at 2.0.2.

## Build and test

From the repository root:

```sh
cargo test
cargo build --bin msengine
uv --project harness sync
uv --project harness run pytest
```

## Generate a board

```sh
mkdir -p runs
cargo run --bin msengine -- generate \
  --seed 1 --rows 9 --cols 9 --mines 10 \
  --out runs/beginner.json
```

`--level beginner|intermediate|expert` fills in the modern sizes (9×9/10, 16×16/40, 16×30/99) so the three flags can be omitted; an explicit `--rows`, `--cols`, or `--mines` still wins one by one.

Mines are drawn uniformly from the cells outside the first click's neighborhood, so that click is an opening. The default click is the center cell, `((rows + 1) / 2, (cols + 1) / 2)`. Generation tries up to 5000 layouts. A board is kept only when every later position still has a forced open. `gold` is one such path: open only, no flags, and at each step the forced open with the lowest coordinate.

```sh
cargo run --bin msengine -- demo
cargo run --bin msengine -- play runs/beginner.json
```

`demo` prints the textbook 3×3. `play` walks `gold` and prints each board.

On Windows the built binary is `target\debug\msengine.exe`. `cargo run --bin msengine --` is the portable form.

## Evaluate a model

Copy `harness/.env.example` to `harness/.env` and set:

- `MS_API_BASE`, the API root ending in `/v1`
- `MS_MODEL`, the model id
- `MS_API_KEY`, kept on the machine; there is no flag for it

A shell variable overrides the file. A CLI flag overrides both. Requests stream (`stream: true`) and do not send `max_tokens`. The default temperature is 0, with one sample per position.

```sh
cp harness/.env.example harness/.env
uv run --project harness msbench gold runs/beginner.json
uv run --project harness msbench one-step runs/beginner.json
uv run --project harness msbench multi-step runs/beginner.json
uv run --project harness msbench summary runs/<run>.jsonl
```

`one-step` asks once at every position along `gold`. Each item reloads the fixture and replays the gold prefix up to that position. `multi-step` keeps one engine session from the opening until the mine explodes or the board is won. Guesses and rejected moves are recorded and play continues; three straight steps without board progress end the episode. Each model call is a new request containing the fixed prompt and the current board. Earlier replies are not sent back.

Logs are written to `runs/{run_id}-{mode}.jsonl`. Records stream to the file as each move is scored, so a run can be watched or resumed from a partial log. `summary` prints a Traditional Chinese report. Passing exactly two logs also prints a paired bootstrap of the second file minus the first.

## Read a log

```sh
uv run --project harness python viewer/serve.py
```

This prints `http://127.0.0.1:8765/` and waits; nothing opens on its own. Open the URL yourself, or pass `--open-browser` to launch the default browser. The server lists `runs/*.jsonl` and will not serve any other path. `--port` changes the port. Stop it with Ctrl+C.

The page is in English. Each record shows the board before the move and the board after it, the scored action, the label, token counts, and the model text. Up and down arrow keys move between records. A file can also be dropped onto the page.

On the board before the move, yellow is a forced open, red is a forced flag, and blue-gray is a forced chord. Green is the chosen move when it is in that set. Black is the chosen move otherwise. On the board after the move, blue marks cells that changed.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MS_API_BASE` | | API root ending in `/v1`. Also `--base`. |
| `MS_MODEL` | | Model id. Also `--model`. |
| `MS_API_KEY` | | Bearer token. File or environment only. |
| `MS_PROTOCOL` | `instruct` | `instruct` or `thinking`. Logged, not sent as a separate API field. Also `--protocol`. |
| `MS_TEMPERATURE` | `0` | Also `--temperature`. Above 0, `MS_SAMPLES` may be greater than 1. |
| `MS_SAMPLES` | `1` | Draws per item, averaged within the item. Also `--samples`. |
| `MS_MAX_RETRIES` | `2` | Extra attempts after a transport failure. Also `--max-retries`. |
| `MS_TIMEOUT` | `120` | Per-request timeout in seconds. Also `--timeout`. |
| `MS_TRACK` | `adhoc` | Label stored on the run. Also `--track`. |
| `MSENGINE` | | Path to the `msengine` binary. Also `--binary`. |

## Board and actions

Coordinates are 1-based. Row 1 is the top row and column 1 is the left column.

| Glyph | Meaning |
| --- | --- |
| `#` | Hidden |
| `F` | Flag |
| `.` | Opened zero |
| `1`–`8` | Opened count |
| `*` | Exploded mine |

Each request sends the board with one forced tool, `minesweeper_move`. The model must call it exactly once per turn:

```json
{"kind": "open|flag|chord", "row": "<positive integer>", "col": "<positive integer>"}
```

Free text around the call is ignored. Zero calls, two or more, a wrong name, or broken arguments is a format error.

A fixture is the JSON written by `msengine generate`: the mine coordinates, the player-visible text after the opening click, and `gold`. The prompt sent to the model contains the visible text only.

## Scoring

A response that has a body receives exactly one of five labels. A forced move scores 1. The other four score 0. Rates use the number of responses that had a body as the denominator.

| Order | Label | Meaning |
| --- | --- | --- |
| 1 | Format | The reply does not contain exactly one valid `minesweeper_move` call. |
| 2 | Illegal | The engine rejects the action. The board is unchanged. |
| 3 | Supported | The action is in the live forced set. Open, flag, and chord all count. Any forced cell counts, not only the gold cell. |
| 4 | Mine | The action applies and the game is lost. |
| 5 | Lucky | The action applies and the game stays in progress, or is won, but the cell was not safe in every consistent layout. |

A cell is forced when every mine layout consistent with the visible numbers agrees on it: safe in all of them, or a mine in all of them. Player flags are marks and are removed before the solver runs. The mine total completes the local answer: when every mine is already proven, all remaining hidden cells are safe; when the hidden cells exactly fill the quota, they are all mines. Single-cell conclusions are written back and the same rule runs until it adds nothing; that closure stays in the single tier. Subset reasoning is used only when the single-cell closure is empty, enumeration only after that, and the mine-count completion runs on top of whatever the local tiers found. Only a position with no local deduction at all is labeled with the global tier. A frontier block wider than 55 cells discards the board.

The reasoning rate is supported / (supported + lucky + mine). Format and illegal counts are reported beside it. Multi-step also reports the clear rate and the prefix, the number of leading supported steps. A clear counts a win even when the path includes guesses. The episode stops only on a mine, a win, a transport failure, the step cap, or three consecutive steps without board progress. A timeout, HTTP 429, HTTP 5xx, or a response with no body is a transport failure. Transport failures stay outside the five labels, and a multi-step episode that fails this way is left out of the clear rate and the prefix mean.

Each request log stores latency, `finish_reason`, input tokens, output tokens, reasoning tokens, and total tokens. Total tokens are input plus output. Reasoning tokens are already included in the output count. Token counts and latency are reported next to the score and are not added into it.

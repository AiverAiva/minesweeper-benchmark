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

Omit `--count` to write one fixture and print it, as before:

```sh
mkdir -p runs
cargo run --bin msengine -- generate \
  --seed 1 --rows 9 --cols 9 --mines 10 \
  --out runs/beginner.json
```

`--level beginner|intermediate|expert` fills in the modern sizes (9×9/10, 16×16/40, 16×30/99) so the three flags can be omitted; an explicit `--rows`, `--cols`, or `--mines` still wins one by one.

`--count N --dir DIR` tries N consecutive seeds starting at `--seed` (the closed range `S .. S+N-1`). That is try-N, not keep-N boards. `--count` requires `--seed` and `--dir` and cannot be used with `--out`. `--attempts` is the rejection budget for each seed, not for the whole run. Each kept board is `DIR/board-{seed}.json`. A seed that burns its budget writes `DIR/board-{seed}.exhausted` and the run continues; exit 0 means the range finished without a hard error, even if every seed exhausted. `catalog.json` is a last-run report over that range, rewritten after every seed. Do not mix levels in one `--dir`.

```sh
mkdir -p runs/beginner
cargo run --bin msengine -- generate --seed 1 --count 32 --level beginner --dir runs/beginner
```

```sh
mkdir -p runs/intermediate
cargo run --bin msengine -- generate --seed 1 --count 32 --level intermediate --dir runs/intermediate
```

Default `--overwrite skip` leaves an existing `board-{seed}.json` (after a size check) and a matching exhausted marker in place, so a killed run can be resumed with the same command. `--overwrite replace` deletes both names and generates again. A later run with a higher `--attempts` retries only those exhausted markers whose stored budget is smaller.

Expert generation is CPU-bound. Use a release binary. Three successful **debug** generations took 35, 438, and 1666 attempts (about 120 ms per attempt, ≈0.14% keep); that does not bound release.

```sh
cargo run --release --bin msengine -- generate --seed 1 --count 8 --level expert --dir runs/expert
```

PowerShell 5.1 does not expand `board-*.json` into `msbench`. Pass full paths:

```powershell
uv run --project harness msbench gold (Get-ChildItem runs\beginner\board-*.json | ForEach-Object FullName)
```

`msbench continue` looks up `runs/{stem}.json`. For `runs/beginner/board-1.json` that stem is `board-1`, so pass `--fixture runs/beginner/board-1.json` (repeatable).

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
uv run --project harness msbench continue runs/<run>.jsonl --timeout 300
uv run --project harness msbench summary runs/<run>.jsonl
uv run --project harness msbench rescore runs/<run>.jsonl
```

`one-step` asks once at every position along `gold`. Each item reloads the fixture and replays the gold prefix up to that position. `multi-step` keeps one engine session from the opening until the mine explodes or the board is won. Guesses and rejected moves are recorded and play continues; three straight steps without board progress end the episode. Each model call is a new request containing the fixed prompt and the current board. Earlier replies are not sent back. `rescore` reads a finished JSONL log, replays it on the frozen board, and prints the old label beside the label from the current rules. It does not call a model. If the board JSON is not at `runs/{fixture}.json`, pass `--fixture path/to/board.json`, the same way `continue` does.

Logs are written to `runs/{run_id}-{mode}.jsonl`. Records stream to the file as each move is scored. `continue` resumes a multi-step log that stopped on a transport failure (timeout, HTTP 429/5xx, empty connection), a stall (three steps with no board progress), or a crash while the board was still playing. It replays applied moves onto the same frozen board and asks again from the interrupted step. A stall resume starts at the next step with a fresh stall limit. Finished games (mine, clear, step cap) are left as they are. Model, protocol, and temperature come from the log unless you pass `--model`, `--protocol`, or `--temperature`. `--timeout` and `--max-retries` always come from the current flags or env, so a longer timeout can finish a run that hit the default 120s. Default appends new records to the same JSONL. `--out other.jsonl` writes the playable prefix plus the new steps to a new file. If the board JSON is not at `runs/{fixture}.json`, pass `--fixture path/to/board.json`. `summary` treats a retried `step_index` as the last record at that step. Passing exactly two logs also prints a paired bootstrap of the second file minus the first.

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

A response that has a body receives exactly one of six labels. A supported move scores 1. The other five score 0. Rates use the number of responses that had a body as the denominator.

| Order | Label | Meaning |
| --- | --- | --- |
| 1 | Format | The reply does not contain exactly one valid `minesweeper_move` call. |
| 2 | Illegal | The engine rejects the action. The board is unchanged. |
| 3 | Mine | The action applies and the game is lost. |
| 4 | Misflag | The action places a flag on a cell that is not a mine, or removes a flag from a cell that is a mine in every consistent layout. |
| 5 | Supported | The action is safe in every mine layout consistent with the visible numbers and the mine total: an open of a necessarily safe cell, a flag of a necessarily mined cell, removal of a flag from a necessarily safe cell, or a chord whose every opened cell is necessarily safe. |
| 6 | Lucky | The action applies, the game is not lost, and none of the supported cases apply. Removing a flag from a cell that is not proved either way is lucky. |

The exact judge uses only the player-visible board. Flags are marks and are removed before the search. Unknown cells split into the frontier (adjacent to an opened number) and the interior. Each frontier component is enumerated, and a solution is kept only when the mines still to place fit in the interior. A cell is safe or a mine when every kept layout agrees. One agreeing layout on each side is enough to leave it undecided; layouts are not weighted. The search writes each assignment in place and undoes it, and it stops after 1,000,000 nodes.

A frontier component wider than 55 cells, or a search that passes the node limit, does not produce a partial set. The mine total couples every component, so the whole step falls back to the oracle `forced()` set. The log records `exact_status` as `ok`, `enum_limit`, or `node_limit`. The run keeps the model reply. Gold paths still follow `forced()`, which stops at the first tier that finds any cell and then applies only the two mine-count completions (every mine already proved, or the unknown cells exactly fill the quota).

`tier_needed` is set only for a supported open, a supported flag of a mine, or a supported chord. A chord takes the deepest tier among the cells it opens. An open of a mine, a flag on a safe cell, an unflag, and every other move log `tier_needed` as null. The tiers are tested separately on the original board, so a hit in single does not skip the later tiers. Summary tables group every move by the position's `position_tier` (the shallowest tier that found any forced cell before the move). `tier_needed` is a separate distribution over supported steps.

The chance baseline is the mean of safe opens divided by hidden cells. New logs store that safe-open count in `exact_open_count` from the exact judge, or from the oracle when `exact_status` is not `ok`. Older logs stored only the oracle `forced_open_count`, so the baseline moves when a log is rescored.

The reasoning rate is supported / (supported + lucky + mine + misflag). Format and illegal counts are reported beside it. Multi-step also reports the clear rate and the prefix, the number of leading supported steps. A clear counts a win even when the path includes guesses. The episode stops only on a mine, a win, a transport failure, the step cap, or three consecutive steps without board progress. A timeout, HTTP 429, HTTP 5xx, or a response with no body is a transport failure. Transport failures stay outside the six labels, and a multi-step episode that fails this way is left out of the clear rate and the prefix mean. `rescore` reprints labels for a log that was scored before this rule. It compares every replayed board with the log and prints a total for each `prompt_version` and `board_format`, including how many steps moved between lucky and supported.

Each request log stores latency, `finish_reason`, input tokens, output tokens, reasoning tokens, and total tokens. Total tokens are input plus output. Reasoning tokens are already included in the output count. Token counts and latency are reported next to the score and are not added into it.

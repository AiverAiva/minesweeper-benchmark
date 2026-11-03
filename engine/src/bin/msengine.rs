//! Text view of the engine. The library has no window; this prints the board.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use minesweeper_engine::{
    generate, generate_batch, play_forced, replay, run_session, Action, BatchEvent, BatchRequest,
    CellPos, Fixture, Game, GenerateRequest, Overwrite, Status, DEFAULT_ATTEMPTS,
};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("generate") => cmd_generate(&args.collect::<Vec<_>>()),
        Some("play") => cmd_play(&args.collect::<Vec<_>>()),
        Some("demo") => cmd_demo(),
        Some("session") => cmd_session(&args.collect::<Vec<_>>()),
        Some("help") | Some("--help") | Some("-h") | None => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown command {other}");
            print_help();
            ExitCode::from(2)
        }
    }
}

fn print_help() {
    println!(
        "\
msengine generate [--seed N] [--count N] [--dir DIR] [--overwrite skip|replace] [--level beginner|intermediate|expert] [--rows R --cols C --mines M] [--attempts N] [--out file.json]
msengine play file.json
msengine demo
msengine session"
    );
}

fn cmd_session(args: &[String]) -> ExitCode {
    if !args.is_empty() {
        eprintln!("session reads JSON lines on stdin and takes no arguments");
        print_help();
        return ExitCode::from(2);
    }
    match run_session() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}

fn cmd_generate(args: &[String]) -> ExitCode {
    let parsed = match parse_generate(args) {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("{error}");
            print_help();
            return ExitCode::from(2);
        }
    };
    if let Some(count) = parsed.count {
        let Some(dir) = parsed.dir else {
            eprintln!("--count requires --dir");
            return ExitCode::from(1);
        };
        return cmd_generate_batch(parsed.request, count, dir, parsed.overwrite);
    }
    let out = parsed.out;
    match generate(&parsed.request) {
        Ok(fixture) => {
            print!("{}", fixture.initial_text);
            println!(
                "seed: {}\nattempts: {}\nbbbv: {}\nmax_tier: {}\ngold_steps: {}",
                fixture.seed,
                fixture.attempts,
                fixture.bbbv,
                fixture.max_tier,
                fixture.gold.len()
            );
            if let Some(path) = out {
                match serde_json::to_string_pretty(&fixture) {
                    Ok(json) => {
                        if let Err(error) = fs::write(&path, json + "\n") {
                            eprintln!("could not write {path}: {error}");
                            return ExitCode::from(1);
                        }
                        println!("wrote {path}");
                    }
                    Err(error) => {
                        eprintln!("could not encode the fixture: {error}");
                        return ExitCode::from(1);
                    }
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}

fn cmd_generate_batch(
    parsed: GenerateRequest,
    count: u32,
    dir: String,
    overwrite: Option<Overwrite>,
) -> ExitCode {
    let overwrite = overwrite.unwrap_or(Overwrite::Skip);
    let request = BatchRequest::new(
        parsed.seed,
        count,
        parsed.rows,
        parsed.cols,
        parsed.mines,
        PathBuf::from(&dir),
    )
    .attempts(parsed.max_attempts)
    .overwrite(overwrite);
    match generate_batch(&request, print_batch_event) {
        Ok(report) => {
            println!(
                "kept: {}\nexhausted: {}\nskipped: {}\nwrote {dir}",
                report.catalog.kept, report.catalog.exhausted, report.catalog.skipped
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}

fn print_batch_event(event: &BatchEvent<'_>) {
    match event {
        BatchEvent::Started { index, total, seed } => {
            eprintln!("generating {index}/{total} seed={seed}");
        }
        BatchEvent::Kept {
            seed,
            fixture,
            elapsed,
            ..
        } => {
            eprintln!(
                "seed={seed} kept attempts={} bbbv={} max_tier={} gold_steps={} {}",
                fixture.attempts,
                fixture.bbbv,
                fixture.max_tier,
                fixture.gold.len(),
                format_elapsed(*elapsed)
            );
        }
        BatchEvent::Exhausted {
            seed,
            attempts,
            elapsed,
            ..
        } => {
            eprintln!(
                "seed={seed} exhausted attempts={attempts} {}",
                format_elapsed(*elapsed)
            );
        }
        BatchEvent::Skipped { seed, path, .. } => {
            eprintln!("seed={seed} skipped exists={}", path.display());
        }
    }
}

fn format_elapsed(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs_f64();
    if seconds < 120.0 {
        format!("{seconds:.1}s")
    } else {
        let total = elapsed.as_secs();
        let minutes = total / 60;
        let secs = total % 60;
        format!("{minutes}m{secs:02}s")
    }
}

#[derive(Debug)]
struct GenerateArgs {
    request: GenerateRequest,
    out: Option<String>,
    count: Option<u32>,
    dir: Option<String>,
    overwrite: Option<Overwrite>,
}

fn parse_generate(args: &[String]) -> Result<GenerateArgs, String> {
    let mut seed = None;
    let mut level = None;
    let mut rows = None;
    let mut cols = None;
    let mut mines = None;
    let mut attempts = None;
    let mut out = None;
    let mut count = None;
    let mut dir = None;
    let mut overwrite = None;
    let mut index = 0;
    while index < args.len() {
        let key = &args[index];
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("{key} needs a value"))?;
        match key.as_str() {
            "--seed" => seed = Some(parse_u64(key, value)?),
            "--level" => level = Some(parse_level(key, value)?),
            "--rows" => rows = Some(parse_u32(key, value)?),
            "--cols" => cols = Some(parse_u32(key, value)?),
            "--mines" => mines = Some(parse_u32(key, value)?),
            "--attempts" => attempts = Some(parse_u32(key, value)?),
            "--out" => out = Some(value.clone()),
            "--count" => count = Some(parse_u32(key, value)?),
            "--dir" => dir = Some(value.clone()),
            "--overwrite" => overwrite = Some(parse_overwrite(key, value)?),
            other => return Err(format!("unknown argument {other}")),
        }
        index += 2;
    }
    let seed_explicit = seed.is_some();
    let seed = seed.unwrap_or_else(rand::random::<u64>);
    // A level fills rows/cols/mines; explicit flags win one by one.
    let (preset_rows, preset_cols, preset_mines) = level.unwrap_or((None, None, None));
    let rows = rows
        .or(preset_rows)
        .ok_or("--rows is required (or pick a --level)")?;
    let cols = cols
        .or(preset_cols)
        .ok_or("--cols is required (or pick a --level)")?;
    let mines = mines
        .or(preset_mines)
        .ok_or("--mines is required (or pick a --level)")?;
    if count.is_none() {
        if dir.is_some() {
            return Err("--dir requires --count".into());
        }
        if overwrite.is_some() {
            return Err("--overwrite requires --count".into());
        }
    }
    if let Some(0) = count {
        return Err("--count must be at least 1".into());
    }
    if count.is_some() && !seed_explicit {
        return Err("--count requires --seed".into());
    }
    if count.is_some() && dir.is_none() {
        return Err("--count requires --dir".into());
    }
    if count.is_some() && out.is_some() {
        return Err("--count cannot be used with --out".into());
    }
    if let Some(n) = count {
        if seed.checked_add(u64::from(n) - 1).is_none() {
            return Err("seed range overflows u64".into());
        }
    }
    let mut request = GenerateRequest::new(seed, rows, cols, mines);
    if let Some(attempts) = attempts {
        request = request.attempts(attempts);
    } else {
        request = request.attempts(DEFAULT_ATTEMPTS);
    }
    Ok(GenerateArgs {
        request,
        out,
        count,
        dir,
        overwrite,
    })
}

fn parse_overwrite(key: &str, value: &str) -> Result<Overwrite, String> {
    match value {
        "skip" => Ok(Overwrite::Skip),
        "replace" => Ok(Overwrite::Replace),
        _ => Err(format!("{key} expects skip or replace, got {value}")),
    }
}

/// Modern Minesweeper presets, as (rows, cols, mines).
fn parse_level(key: &str, value: &str) -> Result<(Option<u32>, Option<u32>, Option<u32>), String> {
    let preset = match value.to_ascii_lowercase().as_str() {
        "beginner" => (9, 9, 10),
        "intermediate" => (16, 16, 40),
        "expert" => (16, 30, 99),
        _ => {
            return Err(format!(
                "{key} expects beginner, intermediate, or expert, got {value}"
            ))
        }
    };
    Ok((Some(preset.0), Some(preset.1), Some(preset.2)))
}

fn parse_u32(key: &str, value: &str) -> Result<u32, String> {
    value
        .parse()
        .map_err(|_| format!("{key} expects an integer, got {value}"))
}

fn parse_u64(key: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("{key} expects an integer, got {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| word.to_string()).collect()
    }

    #[test]
    fn levels_map_to_modern_sizes() {
        let beginner = parse_generate(&args(&["--level", "beginner"])).unwrap();
        assert_eq!(
            (
                beginner.request.rows,
                beginner.request.cols,
                beginner.request.mines
            ),
            (9, 9, 10)
        );
        let intermediate = parse_generate(&args(&["--level", "intermediate"])).unwrap();
        assert_eq!(
            (
                intermediate.request.rows,
                intermediate.request.cols,
                intermediate.request.mines
            ),
            (16, 16, 40)
        );
        let expert = parse_generate(&args(&["--level", "EXPERT"])).unwrap();
        assert_eq!(
            (
                expert.request.rows,
                expert.request.cols,
                expert.request.mines
            ),
            (16, 30, 99)
        );
    }

    #[test]
    fn explicit_flags_override_the_level_one_by_one() {
        let parsed = parse_generate(&args(&["--level", "beginner", "--mines", "3"])).unwrap();
        assert_eq!(
            (
                parsed.request.rows,
                parsed.request.cols,
                parsed.request.mines
            ),
            (9, 9, 3)
        );
    }

    #[test]
    fn without_level_or_sizes_it_errors() {
        assert!(parse_generate(&args(&["--level", "easy"])).is_err());
        assert!(parse_generate(&args(&[])).is_err());
    }

    #[test]
    fn count_without_seed_or_dir_errors() {
        assert_eq!(
            parse_generate(&args(&[
                "--count", "3", "--dir", "X", "--level", "beginner",
            ]))
            .unwrap_err(),
            "--count requires --seed"
        );
        assert_eq!(
            parse_generate(&args(&[
                "--count", "3", "--seed", "1", "--level", "beginner",
            ]))
            .unwrap_err(),
            "--count requires --dir"
        );
    }

    #[test]
    fn count_cannot_be_used_with_out() {
        assert_eq!(
            parse_generate(&args(&[
                "--count",
                "1",
                "--seed",
                "1",
                "--dir",
                "X",
                "--out",
                "board.json",
                "--level",
                "beginner",
            ]))
            .unwrap_err(),
            "--count cannot be used with --out"
        );
    }

    #[test]
    fn dir_and_overwrite_require_count() {
        assert_eq!(
            parse_generate(&args(&["--dir", "X", "--level", "beginner"])).unwrap_err(),
            "--dir requires --count"
        );
        assert_eq!(
            parse_generate(&args(&["--overwrite", "skip", "--level", "beginner",])).unwrap_err(),
            "--overwrite requires --count"
        );
    }

    #[test]
    fn overwrite_replace_parses() {
        let parsed = parse_generate(&args(&[
            "--seed",
            "1",
            "--count",
            "1",
            "--dir",
            "X",
            "--level",
            "beginner",
            "--overwrite",
            "replace",
        ]))
        .unwrap();
        assert_eq!(parsed.overwrite, Some(Overwrite::Replace));
        assert_eq!(
            parse_generate(&args(&[
                "--seed",
                "1",
                "--count",
                "1",
                "--dir",
                "X",
                "--level",
                "beginner",
                "--overwrite",
                "skip",
            ]))
            .unwrap()
            .overwrite,
            Some(Overwrite::Skip)
        );
    }

    #[test]
    fn unknown_overwrite_value_errors() {
        assert!(parse_generate(&args(&[
            "--seed",
            "1",
            "--count",
            "1",
            "--dir",
            "X",
            "--level",
            "beginner",
            "--overwrite",
            "force",
        ]))
        .is_err());
        assert!(parse_generate(&args(&[
            "--seed",
            "1",
            "--count",
            "1",
            "--dir",
            "X",
            "--level",
            "beginner",
            "--overwrite",
            "Replace",
        ]))
        .is_err());
    }

    #[test]
    fn count_with_level_fills_modern_sizes() {
        let parsed = parse_generate(&args(&[
            "--count", "3", "--seed", "1", "--level", "beginner", "--dir", "X",
        ]))
        .unwrap();
        assert_eq!(
            (
                parsed.request.rows,
                parsed.request.cols,
                parsed.request.mines
            ),
            (9, 9, 10)
        );
        assert_eq!(parsed.count, Some(3));
        assert_eq!(parsed.dir.as_deref(), Some("X"));
        assert_eq!(parsed.request.seed, 1);
        assert!(parsed.overwrite.is_none());
    }

    #[test]
    fn count_zero_and_overflowing_seed_range_error() {
        assert_eq!(
            parse_generate(&args(&[
                "--count", "0", "--seed", "1", "--dir", "X", "--level", "beginner",
            ]))
            .unwrap_err(),
            "--count must be at least 1"
        );
        assert_eq!(
            parse_generate(&args(&[
                "--count",
                "2",
                "--seed",
                "18446744073709551615",
                "--dir",
                "X",
                "--level",
                "beginner",
            ]))
            .unwrap_err(),
            "seed range overflows u64"
        );
    }
}

fn cmd_play(args: &[String]) -> ExitCode {
    let Some(path) = args.first() else {
        eprintln!("play needs a fixture path");
        print_help();
        return ExitCode::from(2);
    };
    if args.len() != 1 {
        eprintln!("play takes one fixture path");
        return ExitCode::from(2);
    }
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("could not read {path}: {error}");
            return ExitCode::from(1);
        }
    };
    let fixture: Fixture = match serde_json::from_str(&text) {
        Ok(fixture) => fixture,
        Err(error) => {
            eprintln!("could not parse {path}: {error}");
            return ExitCode::from(1);
        }
    };
    if let Err(error) = replay(&fixture) {
        eprintln!("{error}");
        return ExitCode::from(1);
    }
    print_path(&fixture);
    ExitCode::SUCCESS
}

fn print_path(fixture: &Fixture) {
    println!(
        "first click: {}",
        Action::open(fixture.first_click.row, fixture.first_click.col)
    );
    print!("{}", fixture.initial_text);
    for (index, step) in fixture.gold.iter().enumerate() {
        println!("step {}  {}  tier={}", index + 1, step.action, step.tier);
        print!("{}", step.text_after);
    }
}

fn cmd_demo() -> ExitCode {
    println!("textbook 3x3, mines at row=1 col=1 and row=3 col=3");
    let mut game = match Game::from_mines(3, 3, &[CellPos::new(1, 1), CellPos::new(3, 3)]) {
        Ok(game) => game,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(1);
        }
    };
    let opening = Action::open(1, 3);
    if let Err(error) = game.apply(opening) {
        eprintln!("{error}");
        return ExitCode::from(1);
    }
    println!("first click: {opening}");
    print!("{}", game.player().render());
    match play_forced(&mut game) {
        Ok(gold) => {
            for (index, step) in gold.iter().enumerate() {
                println!("step {}  {}  tier={}", index + 1, step.action, step.tier);
                print!("{}", step.text_after);
            }
            if game.player().status() != Status::Won {
                eprintln!("demo ended {}", game.player().status());
                return ExitCode::from(1);
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}

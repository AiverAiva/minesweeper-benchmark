//! Spawns the real `msengine` via `CARGO_BIN_EXE_msengine`.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir() -> PathBuf {
    let pid = std::process::id();
    let thread_id = format!("{:?}", std::thread::current().id());
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = env::temp_dir().join(format!("msengine-batch-{pid}-{thread_id}-{nanos}"));
    if path.exists() {
        fs::remove_dir_all(&path).unwrap();
    }
    fs::create_dir(&path).unwrap();
    path
}

#[test]
fn batch_all_exhausted_exits_zero() {
    let dir = temp_dir();
    let output = Command::new(env!("CARGO_BIN_EXE_msengine"))
        .arg("generate")
        .arg("--seed")
        .arg("1")
        .arg("--count")
        .arg("3")
        .arg("--rows")
        .arg("5")
        .arg("--cols")
        .arg("5")
        .arg("--mines")
        .arg("2")
        .arg("--attempts")
        .arg("0")
        .arg("--dir")
        .arg(&dir)
        .output()
        .expect("spawn msengine generate");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let code = output.status.code();
    let ok = code == Some(0)
        && stdout.contains("kept: 0")
        && stdout.contains("exhausted: 3")
        && dir.join("catalog.json").is_file()
        && dir.join("board-1.exhausted").is_file()
        && dir.join("board-2.exhausted").is_file()
        && dir.join("board-3.exhausted").is_file()
        && !dir.join("board-1.json").exists()
        && !dir.join("board-2.json").exists()
        && !dir.join("board-3.json").exists();
    let _ = fs::remove_dir_all(&dir);
    assert!(ok, "exit={code:?}\nstdout={stdout}\nstderr={stderr}");
}

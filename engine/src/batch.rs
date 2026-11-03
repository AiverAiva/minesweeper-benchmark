//! Consecutive-seed no-guess generation.
//!
//! Each seed calls [`generate`] independently. Existing `board-{seed}.json`
//! files and matching exhausted markers are skipped by default so a long run
//! can resume.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::fixture::Fixture;
use crate::game::CellPos;
use crate::generate::{generate, GenerateError, GenerateRequest, DEFAULT_ATTEMPTS};
use crate::oracle::{Tier, ORACLE_VERSION};

pub const CATALOG_KIND: &str = "msengine.catalog";
pub const EXHAUSTED_KIND: &str = "msengine.exhausted";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchRequest {
    pub start_seed: u64,
    pub count: u32,
    pub rows: u32,
    pub cols: u32,
    pub mines: u32,
    pub max_attempts: u32,
    pub first_click: Option<CellPos>,
    pub dir: PathBuf,
    pub overwrite: Overwrite,
}

impl BatchRequest {
    pub fn new(
        start_seed: u64,
        count: u32,
        rows: u32,
        cols: u32,
        mines: u32,
        dir: PathBuf,
    ) -> Self {
        Self {
            start_seed,
            count,
            rows,
            cols,
            mines,
            max_attempts: DEFAULT_ATTEMPTS,
            first_click: None,
            dir,
            overwrite: Overwrite::Skip,
        }
    }

    pub fn attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    pub fn first_click(mut self, click: CellPos) -> Self {
        self.first_click = Some(click);
        self
    }

    pub fn overwrite(mut self, overwrite: Overwrite) -> Self {
        self.overwrite = overwrite;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Overwrite {
    Skip,
    Replace,
}

#[derive(Debug)]
pub enum BatchEvent<'a> {
    /// Emitted only immediately before `generate()` runs.
    Started { index: u32, total: u32, seed: u64 },
    Skipped {
        index: u32,
        total: u32,
        seed: u64,
        path: &'a Path,
    },
    Kept {
        index: u32,
        total: u32,
        seed: u64,
        fixture: &'a Fixture,
        path: &'a Path,
        elapsed: Duration,
    },
    Exhausted {
        index: u32,
        total: u32,
        seed: u64,
        attempts: u32,
        path: &'a Path,
        elapsed: Duration,
    },
}

/// In-memory result of a completed range. `catalog` is what was last written to catalog.json.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchReport {
    pub catalog: Catalog,
}

/// On-disk catalog.json. serde of this type is the file, pretty + trailing newline.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Catalog {
    pub kind: String,
    pub oracle_version: String,
    pub rows: u32,
    pub cols: u32,
    pub mines: u32,
    pub max_attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_click: Option<CellPos>,
    pub start_seed: u64,
    pub count: u32,
    pub overwrite: Overwrite,
    pub kept: u32,
    pub exhausted: u32,
    pub skipped: u32,
    pub entries: Vec<CatalogEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub seed: u64,
    #[serde(flatten)]
    pub outcome: SeedOutcome,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum SeedOutcome {
    Kept {
        attempts: u32,
        bbbv: u32,
        max_tier: Tier,
        gold_steps: usize,
        path: PathBuf,
    },
    Exhausted {
        attempts: u32,
        path: PathBuf,
    },
    Skipped {
        path: PathBuf,
    },
}

#[derive(Debug)]
pub enum BatchError {
    Count,
    SeedRange,
    Dir {
        path: PathBuf,
        source: io::Error,
    },
    Write {
        seed: Option<u64>,
        path: PathBuf,
        source: io::Error,
    },
    Encode {
        seed: Option<u64>,
        source: serde_json::Error,
    },
    Generate {
        seed: u64,
        source: GenerateError,
    },
    Mismatch {
        seed: u64,
        path: PathBuf,
        reason: String,
    },
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BatchError::Count => write!(f, "count must be at least 1"),
            BatchError::SeedRange => write!(f, "seed range overflows u64"),
            BatchError::Dir { path, source } => {
                write!(f, "could not create {}: {source}", path.display())
            }
            BatchError::Write {
                seed: Some(seed),
                path,
                source,
            } => write!(
                f,
                "could not write {} (seed {seed}): {source}",
                path.display()
            ),
            BatchError::Write {
                seed: None,
                path,
                source,
            } => write!(f, "could not write {}: {source}", path.display()),
            BatchError::Encode {
                seed: Some(seed),
                source,
            } => write!(f, "could not encode seed {seed}: {source}"),
            BatchError::Encode { seed: None, source } => {
                write!(f, "could not encode the catalog: {source}")
            }
            BatchError::Generate { seed, source } => write!(f, "seed {seed}: {source}"),
            BatchError::Mismatch { seed, path, reason } => write!(
                f,
                "existing {} (seed {seed}) does not match this request: {reason}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for BatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BatchError::Dir { source, .. } | BatchError::Write { source, .. } => Some(source),
            BatchError::Encode { source, .. } => Some(source),
            BatchError::Generate { source, .. } => Some(source),
            BatchError::Count | BatchError::SeedRange | BatchError::Mismatch { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ExhaustedMarker {
    kind: String,
    oracle_version: String,
    seed: u64,
    rows: u32,
    cols: u32,
    mines: u32,
    max_attempts: u32,
    attempts: u32,
}

pub fn generate_batch(
    request: &BatchRequest,
    mut on_event: impl FnMut(&BatchEvent<'_>),
) -> Result<BatchReport, BatchError> {
    if request.count < 1 {
        return Err(BatchError::Count);
    }
    if request
        .start_seed
        .checked_add(u64::from(request.count) - 1)
        .is_none()
    {
        return Err(BatchError::SeedRange);
    }

    fs::create_dir_all(&request.dir).map_err(|source| BatchError::Dir {
        path: request.dir.clone(),
        source,
    })?;
    clean_stale_tmps(&request.dir)?;

    let mut catalog = Catalog {
        kind: CATALOG_KIND.to_string(),
        oracle_version: ORACLE_VERSION.to_string(),
        rows: request.rows,
        cols: request.cols,
        mines: request.mines,
        max_attempts: request.max_attempts,
        first_click: request.first_click,
        start_seed: request.start_seed,
        count: request.count,
        overwrite: request.overwrite,
        kept: 0,
        exhausted: 0,
        skipped: 0,
        entries: Vec::new(),
    };

    for index in 1..=request.count {
        let seed = request.start_seed + u64::from(index - 1);
        process_seed(request, index, seed, &mut catalog, &mut on_event)?;
    }

    Ok(BatchReport { catalog })
}

fn process_seed(
    request: &BatchRequest,
    index: u32,
    seed: u64,
    catalog: &mut Catalog,
    on_event: &mut impl FnMut(&BatchEvent<'_>),
) -> Result<(), BatchError> {
    let total = request.count;
    let rel_json = PathBuf::from(format!("board-{seed}.json"));
    let rel_exh = PathBuf::from(format!("board-{seed}.exhausted"));
    let abs_json = request.dir.join(&rel_json);
    let abs_exh = request.dir.join(&rel_exh);

    remove_if_exists(&tmp_path(&abs_json), Some(seed))?;
    remove_if_exists(&tmp_path(&abs_exh), Some(seed))?;

    match request.overwrite {
        Overwrite::Replace => {
            remove_if_exists(&abs_json, Some(seed))?;
            remove_if_exists(&abs_exh, Some(seed))?;
        }
        Overwrite::Skip => {
            if abs_json.exists() {
                check_fixture_shape(&abs_json, seed, request)?;
                on_event(&BatchEvent::Skipped {
                    index,
                    total,
                    seed,
                    path: &rel_json,
                });
                return record(
                    catalog,
                    &request.dir,
                    CatalogEntry {
                        seed,
                        outcome: SeedOutcome::Skipped { path: rel_json },
                    },
                );
            }
            if abs_exh.exists() && exhausted_marker_covers(&abs_exh, seed, request)? {
                on_event(&BatchEvent::Skipped {
                    index,
                    total,
                    seed,
                    path: &rel_exh,
                });
                return record(
                    catalog,
                    &request.dir,
                    CatalogEntry {
                        seed,
                        outcome: SeedOutcome::Skipped { path: rel_exh },
                    },
                );
            }
        }
    }

    on_event(&BatchEvent::Started { index, total, seed });
    let began = Instant::now();
    let mut gen = GenerateRequest::new(seed, request.rows, request.cols, request.mines)
        .attempts(request.max_attempts);
    if let Some(click) = request.first_click {
        gen = gen.first_click(click);
    }
    match generate(&gen) {
        Ok(fixture) => {
            let elapsed = began.elapsed();
            let json = encode_pretty(&fixture, Some(seed))?;
            atomic_write(&abs_json, &json, Some(seed))?;
            remove_if_exists(&abs_exh, Some(seed))?;
            on_event(&BatchEvent::Kept {
                index,
                total,
                seed,
                fixture: &fixture,
                path: &rel_json,
                elapsed,
            });
            record(
                catalog,
                &request.dir,
                CatalogEntry {
                    seed,
                    outcome: SeedOutcome::Kept {
                        attempts: fixture.attempts,
                        bbbv: fixture.bbbv,
                        max_tier: fixture.max_tier,
                        gold_steps: fixture.gold.len(),
                        path: rel_json,
                    },
                },
            )
        }
        Err(GenerateError::Exhausted { attempts }) => {
            let elapsed = began.elapsed();
            let marker = ExhaustedMarker {
                kind: EXHAUSTED_KIND.to_string(),
                oracle_version: ORACLE_VERSION.to_string(),
                seed,
                rows: request.rows,
                cols: request.cols,
                mines: request.mines,
                max_attempts: request.max_attempts,
                attempts,
            };
            let json = encode_pretty(&marker, Some(seed))?;
            atomic_write(&abs_exh, &json, Some(seed))?;
            on_event(&BatchEvent::Exhausted {
                index,
                total,
                seed,
                attempts,
                path: &rel_exh,
                elapsed,
            });
            record(
                catalog,
                &request.dir,
                CatalogEntry {
                    seed,
                    outcome: SeedOutcome::Exhausted {
                        attempts,
                        path: rel_exh,
                    },
                },
            )
        }
        Err(source) => Err(BatchError::Generate { seed, source }),
    }
}

fn check_fixture_shape(path: &Path, seed: u64, request: &BatchRequest) -> Result<(), BatchError> {
    let text = fs::read_to_string(path).map_err(|source| BatchError::Write {
        seed: Some(seed),
        path: path.to_path_buf(),
        source,
    })?;
    let fixture: Fixture = serde_json::from_str(&text)
        .map_err(|error| mismatch(seed, path, format!("invalid JSON: {error}")))?;
    if fixture.seed != seed {
        return Err(mismatch(seed, path, format!("seed is {}", fixture.seed)));
    }
    if fixture.rows != request.rows {
        return Err(mismatch(seed, path, format!("rows is {}", fixture.rows)));
    }
    if fixture.cols != request.cols {
        return Err(mismatch(seed, path, format!("cols is {}", fixture.cols)));
    }
    if fixture.mines_total != request.mines {
        return Err(mismatch(
            seed,
            path,
            format!("mines_total is {}", fixture.mines_total),
        ));
    }
    Ok(())
}

/// Returns true when skip should honor this marker (do not call generate).
fn exhausted_marker_covers(
    path: &Path,
    seed: u64,
    request: &BatchRequest,
) -> Result<bool, BatchError> {
    let text = fs::read_to_string(path).map_err(|source| BatchError::Write {
        seed: Some(seed),
        path: path.to_path_buf(),
        source,
    })?;
    let marker: ExhaustedMarker = serde_json::from_str(&text)
        .map_err(|error| mismatch(seed, path, format!("invalid JSON: {error}")))?;
    if marker.kind != EXHAUSTED_KIND {
        return Err(mismatch(seed, path, format!("kind is {}", marker.kind)));
    }
    if marker.seed != seed {
        return Err(mismatch(seed, path, format!("seed is {}", marker.seed)));
    }
    if marker.rows != request.rows {
        return Err(mismatch(seed, path, format!("rows is {}", marker.rows)));
    }
    if marker.cols != request.cols {
        return Err(mismatch(seed, path, format!("cols is {}", marker.cols)));
    }
    if marker.mines != request.mines {
        return Err(mismatch(seed, path, format!("mines is {}", marker.mines)));
    }
    Ok(marker.oracle_version == ORACLE_VERSION && marker.max_attempts >= request.max_attempts)
}

fn mismatch(seed: u64, path: &Path, reason: String) -> BatchError {
    BatchError::Mismatch {
        seed,
        path: path.to_path_buf(),
        reason,
    }
}

fn record(catalog: &mut Catalog, dir: &Path, entry: CatalogEntry) -> Result<(), BatchError> {
    catalog.entries.push(entry);
    catalog.kept = 0;
    catalog.exhausted = 0;
    catalog.skipped = 0;
    for entry in &catalog.entries {
        match entry.outcome {
            SeedOutcome::Kept { .. } => catalog.kept += 1,
            SeedOutcome::Exhausted { .. } => catalog.exhausted += 1,
            SeedOutcome::Skipped { .. } => catalog.skipped += 1,
        }
    }
    let json = encode_pretty(catalog, None)?;
    atomic_write(&dir.join("catalog.json"), &json, None)
}

fn encode_pretty<T: Serialize>(value: &T, seed: Option<u64>) -> Result<String, BatchError> {
    serde_json::to_string_pretty(value)
        .map(|json| json + "\n")
        .map_err(|source| BatchError::Encode { seed, source })
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".tmp");
    PathBuf::from(os)
}

fn atomic_write(path: &Path, contents: &str, seed: Option<u64>) -> Result<(), BatchError> {
    let tmp = tmp_path(path);
    fs::write(&tmp, contents).map_err(|source| BatchError::Write {
        seed,
        path: tmp.clone(),
        source,
    })?;
    // Windows rename does not replace an existing destination.
    remove_if_exists(path, seed)?;
    fs::rename(&tmp, path).map_err(|source| BatchError::Write {
        seed,
        path: path.to_path_buf(),
        source,
    })
}

fn remove_if_exists(path: &Path, seed: Option<u64>) -> Result<(), BatchError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(BatchError::Write {
            seed,
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn clean_stale_tmps(dir: &Path) -> Result<(), BatchError> {
    let mut stale = Vec::new();
    let entries = fs::read_dir(dir).map_err(|source| BatchError::Write {
        seed: None,
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| BatchError::Write {
            seed: None,
            path: dir.to_path_buf(),
            source,
        })?;
        if is_stale_tmp(&entry.file_name()) {
            stale.push(entry.path());
        }
    }
    for path in stale {
        remove_if_exists(&path, None)?;
    }
    Ok(())
}

fn is_stale_tmp(name: &std::ffi::OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };
    name == "catalog.json.tmp"
        || (name.starts_with("board-")
            && (name.ends_with(".json.tmp") || name.ends_with(".exhausted.tmp")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::generate;
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn create() -> Self {
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
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn unique_missing(tag: &str) -> PathBuf {
        let pid = std::process::id();
        let thread_id = format!("{:?}", std::thread::current().id());
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        env::temp_dir().join(format!(
            "msengine-batch-missing-{tag}-{pid}-{thread_id}-{nanos}"
        ))
    }

    fn small_batch(start_seed: u64, count: u32, dir: PathBuf) -> BatchRequest {
        BatchRequest::new(start_seed, count, 5, 5, 2, dir).attempts(300)
    }

    #[derive(Debug, PartialEq, Eq)]
    enum Seen {
        Started,
        Skipped,
        Kept,
        Exhausted,
    }

    fn record_events(event: &BatchEvent<'_>, seen: &mut Vec<Seen>) {
        seen.push(match event {
            BatchEvent::Started { .. } => Seen::Started,
            BatchEvent::Skipped { .. } => Seen::Skipped,
            BatchEvent::Kept { .. } => Seen::Kept,
            BatchEvent::Exhausted { .. } => Seen::Exhausted,
        });
    }

    fn board_jsons(dir: &Path) -> Vec<PathBuf> {
        fs::read_dir(dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("board-") && name.ends_with(".json"))
            })
            .collect()
    }

    #[test]
    fn batch_seed_matches_solo_generate() {
        let tmp = TempDir::create();
        let request = small_batch(42, 1, tmp.path.clone());
        let report = generate_batch(&request, |_| {}).unwrap();
        assert_eq!(report.catalog.kept, 1);
        let solo = generate(&GenerateRequest::new(42, 5, 5, 2).attempts(300)).unwrap();
        let text = fs::read_to_string(tmp.path.join("board-42.json")).unwrap();
        let written: Fixture = serde_json::from_str(&text).unwrap();
        assert_eq!(written, solo);
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn replace_rewrites_byte_identical_files() {
        let tmp = TempDir::create();
        let request = small_batch(42, 1, tmp.path.clone()).overwrite(Overwrite::Replace);
        generate_batch(&request, |_| {}).unwrap();
        let first = fs::read(tmp.path.join("board-42.json")).unwrap();
        generate_batch(&request, |_| {}).unwrap();
        let second = fs::read(tmp.path.join("board-42.json")).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn skip_leaves_existing_bytes_and_does_not_start() {
        let tmp = TempDir::create();
        let request = small_batch(42, 1, tmp.path.clone());
        generate_batch(&request, |_| {}).unwrap();
        let first = fs::read(tmp.path.join("board-42.json")).unwrap();
        let mut seen = Vec::new();
        let report = generate_batch(&request, |event| record_events(event, &mut seen)).unwrap();
        assert_eq!(fs::read(tmp.path.join("board-42.json")).unwrap(), first);
        assert_eq!(report.catalog.skipped, request.count);
        assert_eq!(seen, vec![Seen::Skipped]);
    }

    #[test]
    fn zero_attempts_exhausts_every_seed() {
        let tmp = TempDir::create();
        let request = BatchRequest::new(1, 3, 5, 5, 2, tmp.path.clone()).attempts(0);
        let report = generate_batch(&request, |_| {}).unwrap();
        assert_eq!(report.catalog.kept, 0);
        assert_eq!(report.catalog.exhausted, 3);
        assert_eq!(report.catalog.entries.len(), 3);
        assert!(board_jsons(&tmp.path).is_empty());
        for seed in 1..=3 {
            assert!(tmp.path.join(format!("board-{seed}.exhausted")).exists());
        }
        let catalog_text = fs::read_to_string(tmp.path.join("catalog.json")).unwrap();
        assert!(catalog_text.ends_with('\n'));
        let on_disk: Catalog = serde_json::from_str(&catalog_text).unwrap();
        assert_eq!(on_disk, report.catalog);
        assert_eq!(on_disk.kind, CATALOG_KIND);
        assert_eq!(on_disk.oracle_version, ORACLE_VERSION);
    }

    #[test]
    fn skip_does_not_rewrite_exhausted_markers() {
        let tmp = TempDir::create();
        let request = BatchRequest::new(7, 2, 5, 5, 2, tmp.path.clone()).attempts(0);
        generate_batch(&request, |_| {}).unwrap();
        let first_one = fs::read(tmp.path.join("board-7.exhausted")).unwrap();
        let first_two = fs::read(tmp.path.join("board-8.exhausted")).unwrap();
        let mut seen = Vec::new();
        generate_batch(&request, |event| record_events(event, &mut seen)).unwrap();
        assert_eq!(
            fs::read(tmp.path.join("board-7.exhausted")).unwrap(),
            first_one
        );
        assert_eq!(
            fs::read(tmp.path.join("board-8.exhausted")).unwrap(),
            first_two
        );
        assert!(seen.iter().all(|event| *event == Seen::Skipped));
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn a_higher_attempts_budget_retries_an_exhausted_seed() {
        let tmp = TempDir::create();
        let first = BatchRequest::new(42, 1, 5, 5, 2, tmp.path.clone()).attempts(0);
        generate_batch(&first, |_| {}).unwrap();
        assert!(tmp.path.join("board-42.exhausted").exists());
        let mut seen = Vec::new();
        let second = small_batch(42, 1, tmp.path.clone());
        let report = generate_batch(&second, |event| record_events(event, &mut seen)).unwrap();
        assert!(seen.contains(&Seen::Started));
        assert_eq!(report.catalog.kept, 1);
        assert!(tmp.path.join("board-42.json").exists());
        assert!(!tmp.path.join("board-42.exhausted").exists());
    }

    #[test]
    fn skip_of_a_wrong_size_board_is_a_mismatch() {
        let tmp = TempDir::create();
        let mut fixture = generate(&GenerateRequest::new(42, 5, 5, 2).attempts(300)).unwrap();
        fixture.rows = 9;
        fs::write(
            tmp.path.join("board-42.json"),
            serde_json::to_string_pretty(&fixture).unwrap() + "\n",
        )
        .unwrap();
        let request = small_batch(42, 1, tmp.path.clone());
        let error = generate_batch(&request, |_| {}).unwrap_err();
        assert!(matches!(error, BatchError::Mismatch { seed: 42, .. }));
    }

    #[test]
    fn count_zero_and_seed_range_overflow_do_not_create_boards() {
        let missing = unique_missing("count");
        assert!(!missing.exists());
        let error =
            generate_batch(&BatchRequest::new(1, 0, 5, 5, 2, missing.clone()), |_| {}).unwrap_err();
        assert!(matches!(error, BatchError::Count));
        assert!(!missing.exists());

        let missing = unique_missing("range");
        assert!(!missing.exists());
        let error = generate_batch(
            &BatchRequest::new(u64::MAX, 2, 5, 5, 2, missing.clone()),
            |_| {},
        )
        .unwrap_err();
        assert!(matches!(error, BatchError::SeedRange));
        assert!(!missing.exists());
    }

    #[test]
    fn catalog_json_round_trips() {
        let catalog = Catalog {
            kind: CATALOG_KIND.to_string(),
            oracle_version: ORACLE_VERSION.to_string(),
            rows: 9,
            cols: 9,
            mines: 10,
            max_attempts: 5000,
            first_click: None,
            start_seed: 1,
            count: 3,
            overwrite: Overwrite::Skip,
            kept: 1,
            exhausted: 1,
            skipped: 1,
            entries: vec![
                CatalogEntry {
                    seed: 1,
                    outcome: SeedOutcome::Kept {
                        attempts: 2,
                        bbbv: 10,
                        max_tier: Tier::Single,
                        gold_steps: 10,
                        path: PathBuf::from("board-1.json"),
                    },
                },
                CatalogEntry {
                    seed: 2,
                    outcome: SeedOutcome::Exhausted {
                        attempts: 5000,
                        path: PathBuf::from("board-2.exhausted"),
                    },
                },
                CatalogEntry {
                    seed: 3,
                    outcome: SeedOutcome::Skipped {
                        path: PathBuf::from("board-3.json"),
                    },
                },
            ],
        };
        let text = serde_json::to_string_pretty(&catalog).unwrap();
        let parsed: Catalog = serde_json::from_str(&text).unwrap();
        assert_eq!(catalog, parsed);
        assert_eq!(parsed.kind, CATALOG_KIND);
        assert_eq!(parsed.oracle_version, ORACLE_VERSION);
        assert!(!text.contains("first_click"));
    }
}

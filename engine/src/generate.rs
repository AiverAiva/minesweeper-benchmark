//! Seeded mine placement and the no-guess filter.
//!
//! Mines are a uniform sample outside the first click's neighborhood, so that
//! click is an opening. A candidate is kept when every later step still has a
//! forced open. The witness plays the lowest such open and never flags.

use std::collections::HashSet;
use std::fmt;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::fixture::{Fixture, GoldStep};
use crate::game::{index_of, Action, CellPos, Game, Status};
use crate::oracle::{forced, ForceError, ORACLE_VERSION};

pub const DEFAULT_ATTEMPTS: u32 = 5_000;
const MAX_SIDE: u32 = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerateRequest {
    pub seed: u64,
    pub rows: u32,
    pub cols: u32,
    pub mines: u32,
    pub max_attempts: u32,
    pub first_click: Option<CellPos>,
}

impl GenerateRequest {
    pub fn new(seed: u64, rows: u32, cols: u32, mines: u32) -> Self {
        Self {
            seed,
            rows,
            cols,
            mines,
            max_attempts: DEFAULT_ATTEMPTS,
            first_click: None,
        }
    }

    pub fn attempts(mut self, attempts: u32) -> Self {
        self.max_attempts = attempts;
        self
    }

    pub fn first_click(mut self, click: CellPos) -> Self {
        self.first_click = Some(click);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GenerateError {
    BadSize,
    BadClick { row: u32, col: u32 },
    TooManyMines { mines: u32, room: u32 },
    Exhausted { attempts: u32 },
    Inconsistent,
}

impl fmt::Display for GenerateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenerateError::BadSize => {
                write!(
                    f,
                    "board size must be between 1x1 and {MAX_SIDE}x{MAX_SIDE}"
                )
            }
            GenerateError::BadClick { row, col } => {
                write!(f, "first click row={row} col={col} is outside the board")
            }
            GenerateError::TooManyMines { mines, room } => {
                write!(
                    f,
                    "{mines} mines do not fit in the {room} cells outside the opening"
                )
            }
            GenerateError::Exhausted { attempts } => {
                write!(f, "no no-guess board in {attempts} attempts")
            }
            GenerateError::Inconsistent => write!(f, "oracle and hidden board disagreed"),
        }
    }
}

impl std::error::Error for GenerateError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlayError {
    Terminal,
    EnumLimit { cells: usize },
    Inconsistent,
    Stuck,
    Exploded { action: Action },
}

impl fmt::Display for PlayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlayError::Terminal => write!(f, "the game is already over"),
            PlayError::EnumLimit { cells } => {
                write!(f, "frontier of {cells} cells was discarded")
            }
            PlayError::Inconsistent => write!(f, "opened numbers contradict each other"),
            PlayError::Stuck => write!(f, "a position has no forced open"),
            PlayError::Exploded { action } => write!(f, "{action} opened a mine"),
        }
    }
}

impl std::error::Error for PlayError {}

/// Samples until a board clears by forced opens, or the attempt budget ends.
pub fn generate(request: &GenerateRequest) -> Result<Fixture, GenerateError> {
    if request.rows == 0 || request.cols == 0 || request.rows > MAX_SIDE || request.cols > MAX_SIDE
    {
        return Err(GenerateError::BadSize);
    }
    let click = request
        .first_click
        .unwrap_or_else(|| default_click(request.rows, request.cols));
    if index_of(request.rows, request.cols, click.row, click.col).is_err() {
        return Err(GenerateError::BadClick {
            row: click.row,
            col: click.col,
        });
    }
    let protected = protected_cells(request.rows, request.cols, click);
    let room = u64::from(request.rows) * u64::from(request.cols) - protected.len() as u64;
    if u64::from(request.mines) > room {
        return Err(GenerateError::TooManyMines {
            mines: request.mines,
            room: room as u32,
        });
    }

    let mut rng = StdRng::seed_from_u64(request.seed);
    for attempt in 1..=request.max_attempts {
        let mines = place_mines(
            &mut rng,
            request.rows,
            request.cols,
            request.mines,
            &protected,
        );
        let mut game = Game::from_mines(request.rows, request.cols, &mines)
            .map_err(|_| GenerateError::Inconsistent)?;
        game.apply(Action::open(click.row, click.col))
            .map_err(|_| GenerateError::Inconsistent)?;
        if game.player().status() != Status::Playing {
            continue;
        }
        let initial_text = game.player().render();
        let bbbv = game.truth().bbbv();
        let mines = game.truth().mines();
        let gold = match play_forced(&mut game) {
            Ok(gold) if !gold.is_empty() && game.player().status() == Status::Won => gold,
            Ok(_) | Err(PlayError::Stuck | PlayError::EnumLimit { .. }) => continue,
            Err(PlayError::Terminal | PlayError::Inconsistent | PlayError::Exploded { .. }) => {
                return Err(GenerateError::Inconsistent);
            }
        };
        let max_tier = gold
            .iter()
            .map(|step| step.tier)
            .max()
            .expect("gold is non-empty");
        return Ok(Fixture {
            oracle_version: ORACLE_VERSION.to_string(),
            seed: request.seed,
            attempts: attempt,
            rows: request.rows,
            cols: request.cols,
            mines_total: mines.len() as u32,
            first_click: click,
            mines,
            bbbv,
            max_tier,
            initial_text,
            gold,
        });
    }
    Err(GenerateError::Exhausted {
        attempts: request.max_attempts,
    })
}

/// Plays the lowest forced open until the board is clear.
pub fn play_forced(game: &mut Game) -> Result<Vec<GoldStep>, PlayError> {
    if game.player().status() != Status::Playing {
        return Err(PlayError::Terminal);
    }
    let step_limit = game.player().rows() as usize * game.player().cols() as usize;
    let mut gold = Vec::new();
    while game.player().status() == Status::Playing {
        if gold.len() >= step_limit {
            return Err(PlayError::Stuck);
        }
        let judged = match forced(game.player()) {
            Ok(judged) => judged,
            Err(ForceError::EnumLimit { cells }) => return Err(PlayError::EnumLimit { cells }),
            Err(ForceError::Inconsistent) => return Err(PlayError::Inconsistent),
            Err(ForceError::Terminal) => return Err(PlayError::Terminal),
        };
        let Some(tier) = judged.tier else {
            return Err(PlayError::Stuck);
        };
        let Some(action) = judged.opens().min() else {
            return Err(PlayError::Stuck);
        };
        game.apply(action).map_err(|_| PlayError::Inconsistent)?;
        if game.player().status() == Status::Lost {
            return Err(PlayError::Exploded { action });
        }
        gold.push(GoldStep {
            action,
            tier,
            text_after: game.player().render(),
        });
    }
    if game.player().status() != Status::Won {
        return Err(PlayError::Stuck);
    }
    Ok(gold)
}

fn default_click(rows: u32, cols: u32) -> CellPos {
    CellPos::new((rows + 1) / 2, (cols + 1) / 2)
}

fn protected_cells(rows: u32, cols: u32, click: CellPos) -> HashSet<(usize, usize)> {
    let (r, c) = index_of(rows, cols, click.row, click.col).expect("click is on the board");
    let mut cells = HashSet::with_capacity(9);
    for dr in -1i32..=1 {
        for dc in -1i32..=1 {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr >= 0 && nc >= 0 && (nr as u32) < rows && (nc as u32) < cols {
                cells.insert((nr as usize, nc as usize));
            }
        }
    }
    cells
}

fn place_mines(
    rng: &mut StdRng,
    rows: u32,
    cols: u32,
    mines: u32,
    protected: &HashSet<(usize, usize)>,
) -> Vec<CellPos> {
    let mut candidates = Vec::new();
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            if !protected.contains(&(r, c)) {
                candidates.push((r, c));
            }
        }
    }
    candidates.shuffle(rng);
    candidates
        .into_iter()
        .take(mines as usize)
        .map(|(r, c)| CellPos::new(r as u32 + 1, c as u32 + 1))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        replay, Action, CellPos, Fixture, Game, ReplayError, Status, Tier, ORACLE_VERSION,
    };

    fn small() -> GenerateRequest {
        GenerateRequest::new(42, 5, 5, 2).attempts(300)
    }

    #[test]
    fn the_same_seed_returns_the_same_fixture() {
        let first = generate(&small()).unwrap();
        let second = generate(&small()).unwrap();
        assert_eq!(first, second);
        assert!(first.attempts >= 1);
        assert!(!first.gold.is_empty());
        replay(&first).unwrap();
    }

    #[test]
    fn a_fixture_survives_json() {
        let fixture = generate(&small()).unwrap();
        let text = serde_json::to_string(&fixture).unwrap();
        let parsed = serde_json::from_str(&text).unwrap();
        assert_eq!(fixture, parsed);
        replay(&parsed).unwrap();
    }

    #[test]
    fn an_unforced_gold_step_stops_replay() {
        let mut fixture = generate(&small()).unwrap();
        fixture.gold[0].action = Action::open(0, 1);
        assert!(matches!(
            replay(&fixture),
            Err(ReplayError::Unsupported { .. })
        ));
    }

    #[test]
    fn the_opening_neighborhood_limits_the_mine_count() {
        let request = GenerateRequest::new(1, 3, 3, 1);
        assert!(matches!(
            generate(&request),
            Err(GenerateError::TooManyMines { room: 0, .. })
        ));
    }

    #[test]
    fn a_click_off_the_board_is_rejected() {
        let request = GenerateRequest::new(1, 5, 5, 1).first_click(CellPos::new(0, 1));
        assert!(matches!(
            generate(&request),
            Err(GenerateError::BadClick { row: 0, col: 1 })
        ));
    }

    #[test]
    fn textbook_witness_clears_and_replays() {
        let mut game = Game::from_mines(3, 3, &[CellPos::new(1, 1), CellPos::new(3, 3)]).unwrap();
        game.apply(Action::open(1, 3)).unwrap();
        let initial_text = game.player().render();
        assert_eq!(game.truth().bbbv(), 2);
        let gold = play_forced(&mut game).unwrap();
        assert_eq!(game.player().status(), Status::Won);
        assert_eq!(gold[0].action, Action::open(3, 1));
        assert_eq!(gold[0].tier, Tier::Enumerate);
        assert_eq!(
            gold.last().unwrap().text_after,
            "\
status: won
mines_total: 2
flags_left: 2
rows: 3
cols: 3
board:
R01  #1.
R02  121
R03  .1#
"
        );
        let fixture = Fixture {
            oracle_version: ORACLE_VERSION.to_string(),
            seed: 0,
            attempts: 1,
            rows: 3,
            cols: 3,
            mines_total: 2,
            first_click: CellPos::new(1, 3),
            mines: vec![CellPos::new(1, 1), CellPos::new(3, 3)],
            bbbv: 2,
            max_tier: gold.iter().map(|step| step.tier).max().unwrap(),
            initial_text,
            gold,
        };
        replay(&fixture).unwrap();
    }
}

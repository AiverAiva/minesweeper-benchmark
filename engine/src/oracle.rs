//! Frontier oracle with a mine-count completion.
//!
//! A cell is forced when every mine layout consistent with the opened numbers
//! agrees on it. Player flags are marks, not constraints: they are removed
//! before the solver runs, so both hidden and flagged cells count as unknown.
//! Opened cells are safe, so every mine is still unknown and there are exactly
//! `mines_total` of them. The local tiers (`ms_toollib` frontier solvers)
//! never use that total; the global tier does, and only it does.
//!
//! Tiers are tried shallow-first: single cell, subset, then exact enumeration,
//! until one proves a safe cell. Mines proven on the way are marked and carried
//! into the deeper tiers. A single-cell conclusion is written back and the same rule runs again, so a
//! cell reached by repeating that rule stays in the single tier. The global
//! completion runs on top of whatever the local tiers found: when every mine
//! is already proven the rest unknown is safe (G1), and when the unknown cells
//! exactly fill the quota the hidden ones are mines (G2). A position whose
//! first safe cell comes from a local tier keeps that tier; one whose safe
//! cells come only from the completion, or with no local deduction at all, is
//! labeled with the global tier. The completion only adds
//! cells, so every judgment made without it stays valid.
//! A frontier block larger than [`MAX_ENUM_CELLS`] is rejected instead of treated as a guess.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::game::{
    index_of, neighbor_indexes, Action, ActionKind, CellPos, PlayerBoard, Status, Visible,
};

pub const ORACLE_VERSION: &str = "1.1.0";
pub const MAX_ENUM_CELLS: usize = 55;

/// Shallowest deduction that still forces a cell.
/// Declaration order is depth order: single < subset < enumerate < global.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Single,
    Subset,
    Enumerate,
    Global,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Single => "single",
            Tier::Subset => "subset",
            Tier::Enumerate => "enumerate",
            Tier::Global => "global",
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Forced {
    pub tier: Option<Tier>,
    pub actions: Vec<Action>,
}

impl Forced {
    pub fn opens(&self) -> impl Iterator<Item = Action> + '_ {
        self.actions
            .iter()
            .copied()
            .filter(|action| action.kind == ActionKind::Open)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ForceError {
    Terminal,
    EnumLimit { cells: usize },
    Inconsistent,
}

impl fmt::Display for ForceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForceError::Terminal => write!(f, "no deduction is required after the game has ended"),
            ForceError::EnumLimit { cells } => {
                write!(
                    f,
                    "a frontier of {cells} cells exceeds the enumeration limit of {MAX_ENUM_CELLS}"
                )
            }
            ForceError::Inconsistent => write!(f, "opened numbers contradict each other"),
        }
    }
}

impl std::error::Error for ForceError {}

/// Actions supported by the current player view.
pub fn forced(player: &PlayerBoard) -> Result<Forced, ForceError> {
    if player.status() != Status::Playing {
        return Err(ForceError::Terminal);
    }
    let (safe, mines, tier) = deduce(player)?;
    Ok(Forced {
        tier: if safe.is_empty() && mines.is_empty() {
            None
        } else {
            Some(tier)
        },
        actions: actions_from(player, &safe, &mines),
    })
}

/// Tiers run shallow-first until one proves a safe cell. A mine-only result
/// does not stop the search, because the gold path only opens: the mines are
/// marked on the board so deeper tiers and the count completion can use them.
/// The tier is the one that proved the first safe cell.
fn deduce(
    player: &PlayerBoard,
) -> Result<(BTreeSet<CellPos>, BTreeSet<CellPos>, Tier), ForceError> {
    let mut board = solver_board(player);
    let mut known = BTreeSet::new();
    let mut first = None;
    for tier in [Tier::Single, Tier::Subset, Tier::Enumerate] {
        let (safe, mines) = match tier {
            Tier::Single => run_direct(&board)?,
            Tier::Subset => run_subset(&board)?,
            _ => run_enumerate(&board)?,
        };
        if first.is_none() && !(safe.is_empty() && mines.is_empty()) {
            first = Some(tier);
        }
        for &cell in &mines {
            write_mark(&mut board, cell, 11);
        }
        known.extend(mines);
        if !safe.is_empty() {
            let (safe, mines) = extend_global(player, safe, known);
            return Ok((safe, mines, tier));
        }
    }
    let (safe, mines) = extend_global(player, BTreeSet::new(), known);
    if !safe.is_empty() || (first.is_none() && !mines.is_empty()) {
        return Ok((safe, mines, Tier::Global));
    }
    Ok((safe, mines, first.unwrap_or(Tier::Enumerate)))
}

fn run_enumerate(
    board: &Vec<Vec<i32>>,
) -> Result<(BTreeSet<CellPos>, BTreeSet<CellPos>), ForceError> {
    let (coefficients, variables, constants, _, _) = ms_toollib::refresh_matrixs(board);
    let widest = variables.iter().map(Vec::len).max().unwrap_or(0);
    if widest > MAX_ENUM_CELLS {
        return Err(ForceError::EnumLimit { cells: widest });
    }
    Ok(split_cells(ms_toollib::solve_enumerate(
        &coefficients,
        &variables,
        &constants,
    )))
}

/// Mine-count completion over the local answer.
///
/// Opened cells are safe, so the `mines_total` mines are all inside the
/// unknown (hidden or flagged) cells. Two sound rules, one pass, no feedback
/// into the local tiers:
/// G1: every mine already proven, so the rest unknown is safe.
/// G2: unknown cells exactly fill the quota, so each of them is a mine.
/// Only adds cells; positions judged without it keep the same tier.
fn extend_global(
    player: &PlayerBoard,
    mut safe: BTreeSet<CellPos>,
    mut mines: BTreeSet<CellPos>,
) -> (BTreeSet<CellPos>, BTreeSet<CellPos>) {
    let total = player.mines_total() as usize;
    if total == 0 {
        return (safe, mines);
    }
    let unknown = unknown_cells(player);
    if mines.len() == total {
        for cell in unknown {
            if !mines.contains(&cell) {
                safe.insert(cell);
            }
        }
        return (safe, mines);
    }
    if unknown.len() == total {
        mines.extend(unknown);
    }
    (safe, mines)
}

fn unknown_cells(player: &PlayerBoard) -> BTreeSet<CellPos> {
    let mut cells = BTreeSet::new();
    for row in 1..=player.rows() {
        for col in 1..=player.cols() {
            if matches!(
                player.get(row, col).expect("in range"),
                Visible::Hidden | Visible::Flag
            ) {
                cells.insert(CellPos::new(row, col));
            }
        }
    }
    cells
}

fn run_direct(board: &Vec<Vec<i32>>) -> Result<(BTreeSet<CellPos>, BTreeSet<CellPos>), ForceError> {
    let mut marked = board.clone();
    let mut safe = BTreeSet::new();
    let mut mines = BTreeSet::new();
    loop {
        let (found_safe, found_mines) = direct_pass(&marked)?;
        let mut advanced = false;
        for cell in found_safe {
            if safe.insert(cell) {
                write_mark(&mut marked, cell, 12);
                advanced = true;
            }
        }
        for cell in found_mines {
            if mines.insert(cell) {
                write_mark(&mut marked, cell, 11);
                advanced = true;
            }
        }
        if !advanced {
            break;
        }
    }
    Ok((safe, mines))
}

fn direct_pass(
    board: &Vec<Vec<i32>>,
) -> Result<(BTreeSet<CellPos>, BTreeSet<CellPos>), ForceError> {
    let (mut coefficients, mut variables, mut constants, _, _) = ms_toollib::refresh_matrixs(board);
    let mut working = board.to_vec();
    let found = ms_toollib::solve_direct(
        &mut coefficients,
        &mut variables,
        &mut constants,
        &mut working,
    )
    .map_err(|_| ForceError::Inconsistent)?;
    Ok(split_cells(found))
}

fn write_mark(board: &mut Vec<Vec<i32>>, cell: CellPos, value: i32) {
    let row = (cell.row - 1) as usize;
    let col = (cell.col - 1) as usize;
    board[row][col] = value;
}

fn run_subset(board: &Vec<Vec<i32>>) -> Result<(BTreeSet<CellPos>, BTreeSet<CellPos>), ForceError> {
    let (mut coefficients, mut variables, mut constants, _, _) = ms_toollib::refresh_matrixs(board);
    let mut marked = board.to_vec();
    let found = ms_toollib::solve_minus(
        &mut coefficients,
        &mut variables,
        &mut constants,
        &mut marked,
    )
    .map_err(|_| ForceError::Inconsistent)?;
    Ok(split_cells(found))
}

fn split_cells(
    found: (Vec<(usize, usize)>, Vec<(usize, usize)>),
) -> (BTreeSet<CellPos>, BTreeSet<CellPos>) {
    let safe = found.0.into_iter().map(pos_from_zero).collect();
    let mines = found.1.into_iter().map(pos_from_zero).collect();
    (safe, mines)
}

fn solver_board(player: &PlayerBoard) -> Vec<Vec<i32>> {
    let mut board = vec![vec![10; player.cols() as usize]; player.rows() as usize];
    for row in 1..=player.rows() {
        for col in 1..=player.cols() {
            if let Visible::Number(n) = player.get(row, col).expect("in range") {
                let (r, c) = index_of(player.rows(), player.cols(), row, col).expect("in range");
                board[r][c] = i32::from(n);
            }
        }
    }
    board
}

fn pos_from_zero((r, c): (usize, usize)) -> CellPos {
    CellPos::new(r as u32 + 1, c as u32 + 1)
}

fn actions_from(
    player: &PlayerBoard,
    safe: &BTreeSet<CellPos>,
    mines: &BTreeSet<CellPos>,
) -> Vec<Action> {
    let mut actions = BTreeSet::new();
    for cell in safe {
        match player.get(cell.row, cell.col) {
            Some(Visible::Hidden) => {
                actions.insert(Action::open(cell.row, cell.col));
            }
            Some(Visible::Flag) => {
                actions.insert(Action::flag(cell.row, cell.col));
            }
            _ => {}
        }
    }
    for cell in mines {
        if matches!(player.get(cell.row, cell.col), Some(Visible::Hidden)) {
            actions.insert(Action::flag(cell.row, cell.col));
        }
    }
    actions.extend(chords(player, safe, mines));
    actions.into_iter().collect()
}

fn chords(
    player: &PlayerBoard,
    safe: &BTreeSet<CellPos>,
    mines: &BTreeSet<CellPos>,
) -> Vec<Action> {
    let mut found = Vec::new();
    for row in 1..=player.rows() {
        for col in 1..=player.cols() {
            let Visible::Number(number) = player.get(row, col).expect("in range") else {
                continue;
            };
            let (r, c) = index_of(player.rows(), player.cols(), row, col).expect("in range");
            let neighbors = neighbor_indexes(player.rows() as usize, player.cols() as usize, r, c);
            let mut flags = Vec::new();
            let mut hidden = Vec::new();
            for (nr, nc) in neighbors {
                let at = pos_from_zero((nr, nc));
                match player.get(at.row, at.col).expect("in range") {
                    Visible::Flag => flags.push(at),
                    Visible::Hidden => hidden.push(at),
                    Visible::Number(_) | Visible::Exploded => {}
                }
            }
            if flags.len() == usize::from(number)
                && !hidden.is_empty()
                && flags.iter().all(|cell| mines.contains(cell))
                && hidden.iter().all(|cell| safe.contains(cell))
            {
                found.push(Action::chord(row, col));
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Action, CellPos, Game, PlayerBoard, Status, Visible};

    #[test]
    fn a_one_with_a_single_neighbor_is_a_mine() {
        // Opening the 1 leaves exactly one hidden neighbor, and another safe
        // cell is still closed, so the game has not already been won.
        let mut game = Game::from_mines(2, 3, &[CellPos::new(1, 2)]).unwrap();
        game.apply(Action::open(1, 1)).unwrap();
        game.apply(Action::open(2, 1)).unwrap();
        game.apply(Action::open(2, 2)).unwrap();
        game.apply(Action::open(2, 3)).unwrap();
        assert_eq!(game.player().status(), Status::Playing);
        let judged = forced(game.player()).unwrap();
        assert_eq!(judged.tier, Some(Tier::Single));
        assert!(judged.actions.contains(&Action::flag(1, 2)));
        assert!(!judged.actions.contains(&Action::open(1, 2)));
    }

    #[test]
    fn shared_pair_has_no_forced_cell() {
        let mut game = Game::from_mines(2, 2, &[CellPos::new(1, 2)]).unwrap();
        game.apply(Action::open(1, 1)).unwrap();
        game.apply(Action::open(2, 1)).unwrap();
        let forced = forced(game.player()).unwrap();
        assert!(forced.actions.is_empty());
        assert_eq!(forced.tier, None);
        assert_eq!(game.player().status(), Status::Playing);
    }

    #[test]
    fn proven_quota_makes_distant_cells_safe() {
        // G1: the single tier proves the only mine, so the far hidden cell is
        // safe by count even though no number touches it.
        let mut board = visible_board(&["#1..", "11..", "....", "...#"]);
        board.set_mines_total(1);
        let judged = forced(&board).unwrap();
        assert_eq!(judged.tier, Some(Tier::Single));
        assert!(judged.actions.contains(&Action::flag(1, 1)));
        assert!(judged.actions.contains(&Action::open(4, 4)));
    }

    #[test]
    fn quota_filling_unknowns_are_mines() {
        // G2: no number constrains anything, but the two hidden cells exactly
        // fill the quota of two.
        let mut board = visible_board(&["##"]);
        board.set_mines_total(2);
        let judged = forced(&board).unwrap();
        assert_eq!(judged.tier, Some(Tier::Global));
        assert!(judged.actions.contains(&Action::flag(1, 1)));
        assert!(judged.actions.contains(&Action::flag(1, 2)));
    }

    #[test]
    fn textbook_opening_forces_the_safe_corner() {
        let mut game = Game::from_mines(3, 3, &[CellPos::new(1, 1), CellPos::new(3, 3)]).unwrap();
        game.apply(Action::open(1, 3)).unwrap();
        let forced = forced(game.player()).unwrap();
        assert_eq!(forced.tier, Some(Tier::Enumerate));
        assert!(forced.actions.contains(&Action::open(3, 1)));
    }

    #[test]
    fn a_flag_is_not_a_constraint() {
        let mut game = Game::from_mines(3, 3, &[CellPos::new(1, 1), CellPos::new(3, 3)]).unwrap();
        game.apply(Action::open(1, 3)).unwrap();
        game.apply(Action::flag(2, 1)).unwrap();
        let forced = forced(game.player()).unwrap();
        assert!(forced.actions.contains(&Action::open(3, 1)));
    }

    #[test]
    fn repeated_single_cell_steps_stay_in_the_single_tier() {
        // One pass proves (9,5) safe and (8,7) a mine. Substituting those
        // conclusions makes (9,6) a mine and (9,7) safe, still by one equation.
        let board = visible_board(&[
            "##2111#2#",
            "222#11121",
            "..111....",
            ".........",
            ".........",
            ".......11",
            ".....123#",
            ".11112###",
            ".1#1#####",
        ]);
        let judged = forced(&board).unwrap();
        assert_eq!(judged.tier, Some(Tier::Single));
        assert!(judged.actions.contains(&Action::open(9, 5)));
        assert!(judged.actions.contains(&Action::flag(8, 7)));
        assert!(judged.actions.contains(&Action::flag(9, 6)));
        assert!(judged.actions.contains(&Action::open(9, 7)));
    }

    #[test]
    fn a_mine_only_single_step_still_searches_for_an_open() {
        // The single tier only proves (6,1) and (7,2) are mines. The open at
        // (3,3) needs a deeper tier, and the gold path only opens.
        let board = visible_board(&[
            "#########",
            "#########",
            "#########",
            "11111113#",
            "11.....2#",
            "#21....2#",
            "2#1....2#",
            "111....2#",
            ".......1#",
        ]);
        let judged = forced(&board).unwrap();
        assert!(judged.actions.contains(&Action::flag(6, 1)));
        assert!(judged.actions.contains(&Action::flag(7, 2)));
        assert!(judged.actions.contains(&Action::open(3, 3)));
        assert!(judged.tier > Some(Tier::Single));
    }

    fn visible_board(rows: &[&str]) -> PlayerBoard {
        let cells = rows
            .iter()
            .map(|row| {
                row.chars()
                    .map(|glyph| match glyph {
                        '#' => Visible::Hidden,
                        '.' => Visible::Number(0),
                        '1'..='8' => Visible::Number(glyph.to_digit(10).unwrap() as u8),
                        other => panic!("bad glyph {other}"),
                    })
                    .collect()
            })
            .collect();
        PlayerBoard::from_visible(cells)
    }

    #[test]
    fn a_frontier_wider_than_the_enum_cap_is_rejected() {
        // Two hidden rows around a line of numbers. Ends are 1 and the
        // interior is 2, so no single-cell or subset step pins a cell, and
        // the hidden cells stay one block.
        let cols = 30;
        let mut cells = Vec::with_capacity(3);
        cells.push(vec![Visible::Hidden; cols]);
        let mut middle = vec![Visible::Number(2); cols];
        middle[0] = Visible::Number(1);
        middle[cols - 1] = Visible::Number(1);
        cells.push(middle);
        cells.push(vec![Visible::Hidden; cols]);
        let board = PlayerBoard::from_visible(cells);
        match forced(&board) {
            Err(ForceError::EnumLimit { cells }) => assert!(cells > MAX_ENUM_CELLS),
            other => panic!("expected enum limit, got {other:?}"),
        }
    }
}

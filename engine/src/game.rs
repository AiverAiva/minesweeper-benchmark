use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

/// 1-based board coordinate. Row 1 is the top row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CellPos {
    pub row: u32,
    pub col: u32,
}

impl CellPos {
    pub fn new(row: u32, col: u32) -> Self {
        Self { row, col }
    }
}

impl fmt::Display for CellPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "row={} col={}", self.row, self.col)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    Open,
    Flag,
    Chord,
}

/// One player action. `flag` toggles a flag on a hidden cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Action {
    pub kind: ActionKind,
    pub at: CellPos,
}

impl Action {
    pub fn open(row: u32, col: u32) -> Self {
        Self {
            kind: ActionKind::Open,
            at: CellPos::new(row, col),
        }
    }

    pub fn flag(row: u32, col: u32) -> Self {
        Self {
            kind: ActionKind::Flag,
            at: CellPos::new(row, col),
        }
    }

    pub fn chord(row: u32, col: u32) -> Self {
        Self {
            kind: ActionKind::Chord,
            at: CellPos::new(row, col),
        }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            ActionKind::Open => "open",
            ActionKind::Flag => "flag",
            ActionKind::Chord => "chord",
        };
        write!(f, "ACTION: {kind} {}", self.at)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Playing,
    Won,
    Lost,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Status::Playing => "playing",
            Status::Won => "won",
            Status::Lost => "lost",
        })
    }
}

/// What the player can see in one cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visible {
    Hidden,
    Flag,
    /// Opened safe cell. `0` is an empty opening.
    Number(u8),
    Exploded,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlayerBoard {
    rows: u32,
    cols: u32,
    mines_total: u32,
    status: Status,
    exploded: Option<CellPos>,
    cells: Vec<Vec<Visible>>,
}

impl PlayerBoard {
    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn cols(&self) -> u32 {
        self.cols
    }

    pub fn mines_total(&self) -> u32 {
        self.mines_total
    }

    pub fn status(&self) -> Status {
        self.status
    }

    pub fn exploded(&self) -> Option<CellPos> {
        self.exploded
    }

    pub fn get(&self, row: u32, col: u32) -> Option<Visible> {
        let (r, c) = index_of(self.rows, self.cols, row, col).ok()?;
        Some(self.cells[r][c])
    }

    pub fn flag_count(&self) -> u32 {
        self.cells
            .iter()
            .flatten()
            .filter(|cell| matches!(cell, Visible::Flag))
            .count() as u32
    }

    /// `mines_total - flags`. Negative when the player over-flags.
    pub fn flags_left(&self) -> i32 {
        self.mines_total as i32 - self.flag_count() as i32
    }

    /// Test boards whose numbers are the constraint system under study.
    #[cfg(test)]
    pub(crate) fn from_visible(cells: Vec<Vec<Visible>>) -> Self {
        let rows = cells.len() as u32;
        let cols = cells.first().map(Vec::len).unwrap_or(0) as u32;
        assert!(cells.iter().all(|row| row.len() == cols as usize));
        Self {
            rows,
            cols,
            mines_total: 0,
            status: Status::Playing,
            exploded: None,
            cells,
        }
    }
}

/// Hidden mine map. `-1` is a mine, `0` through `8` are neighbor counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Truth {
    rows: u32,
    cols: u32,
    cells: Vec<Vec<i8>>,
}

impl Truth {
    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn cols(&self) -> u32 {
        self.cols
    }

    pub fn get(&self, row: u32, col: u32) -> Option<i8> {
        let (r, c) = index_of(self.rows, self.cols, row, col).ok()?;
        Some(self.cells[r][c])
    }

    pub fn mines(&self) -> Vec<CellPos> {
        let mut mines = Vec::new();
        for (r, row) in self.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                if *cell < 0 {
                    mines.push(CellPos::new(r as u32 + 1, c as u32 + 1));
                }
            }
        }
        mines
    }

    pub fn mine_count(&self) -> u32 {
        self.cells
            .iter()
            .flatten()
            .filter(|cell| **cell < 0)
            .count() as u32
    }

    /// Clicks a no-flag perfect player needs: one per zero-region, plus every
    /// numbered cell that does not touch a zero.
    pub fn bbbv(&self) -> u32 {
        let rows = self.rows as usize;
        let cols = self.cols as usize;
        let mut seen = vec![vec![false; cols]; rows];
        let mut openings = 0u32;
        for r in 0..rows {
            for c in 0..cols {
                if self.cells[r][c] == 0 && !seen[r][c] {
                    openings += 1;
                    let mut stack = vec![(r, c)];
                    seen[r][c] = true;
                    while let Some((r, c)) = stack.pop() {
                        for (nr, nc) in neighbor_indexes(rows, cols, r, c) {
                            if self.cells[nr][nc] == 0 && !seen[nr][nc] {
                                seen[nr][nc] = true;
                                stack.push((nr, nc));
                            }
                        }
                    }
                }
            }
        }
        let mut isolated = 0u32;
        for r in 0..rows {
            for c in 0..cols {
                if self.cells[r][c] > 0
                    && neighbor_indexes(rows, cols, r, c)
                        .into_iter()
                        .all(|(nr, nc)| self.cells[nr][nc] != 0)
                {
                    isolated += 1;
                }
            }
        }
        openings + isolated
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Game {
    truth: Truth,
    player: PlayerBoard,
}

impl Game {
    pub fn from_mines(rows: u32, cols: u32, mines: &[CellPos]) -> Result<Self, GameError> {
        if rows == 0 || cols == 0 {
            return Err(GameError::EmptyBoard);
        }
        let mut seen = HashSet::new();
        let mut mine_at = HashSet::new();
        for mine in mines {
            let (r, c) = index_of(rows, cols, mine.row, mine.col).map_err(|_| {
                GameError::MineOutOfRange {
                    row: mine.row,
                    col: mine.col,
                }
            })?;
            if !seen.insert((r, c)) {
                return Err(GameError::DuplicateMine {
                    row: mine.row,
                    col: mine.col,
                });
            }
            mine_at.insert((r, c));
        }
        let cells = number_grid(rows as usize, cols as usize, &mine_at);
        let truth = Truth { rows, cols, cells };
        let player = PlayerBoard {
            rows,
            cols,
            mines_total: truth.mine_count(),
            status: Status::Playing,
            exploded: None,
            cells: vec![vec![Visible::Hidden; cols as usize]; rows as usize],
        };
        Ok(Self { truth, player })
    }

    pub fn truth(&self) -> &Truth {
        &self.truth
    }

    pub fn player(&self) -> &PlayerBoard {
        &self.player
    }

    pub fn apply(&mut self, action: Action) -> Result<(), ApplyError> {
        if self.player.status != Status::Playing {
            return Err(ApplyError::Terminal);
        }
        let (r, c) = index_of(
            self.player.rows,
            self.player.cols,
            action.at.row,
            action.at.col,
        )?;
        match action.kind {
            ActionKind::Open => self.open_cell(r, c),
            ActionKind::Flag => self.toggle_flag(r, c),
            ActionKind::Chord => self.chord(r, c),
        }
    }

    fn open_cell(&mut self, r: usize, c: usize) -> Result<(), ApplyError> {
        match self.player.cells[r][c] {
            Visible::Hidden => self.reveal(r, c),
            Visible::Flag => Err(ApplyError::Flagged { at: pos_of(r, c) }),
            Visible::Number(_) | Visible::Exploded => {
                Err(ApplyError::AlreadyOpen { at: pos_of(r, c) })
            }
        }
    }

    fn toggle_flag(&mut self, r: usize, c: usize) -> Result<(), ApplyError> {
        match self.player.cells[r][c] {
            Visible::Hidden => {
                self.player.cells[r][c] = Visible::Flag;
                Ok(())
            }
            Visible::Flag => {
                self.player.cells[r][c] = Visible::Hidden;
                Ok(())
            }
            Visible::Number(_) | Visible::Exploded => {
                Err(ApplyError::AlreadyOpen { at: pos_of(r, c) })
            }
        }
    }

    fn chord(&mut self, r: usize, c: usize) -> Result<(), ApplyError> {
        let Visible::Number(number) = self.player.cells[r][c] else {
            return Err(ApplyError::NotANumber { at: pos_of(r, c) });
        };
        let neighbors =
            neighbor_indexes(self.player.rows as usize, self.player.cols as usize, r, c);
        let flags = neighbors
            .iter()
            .filter(|(nr, nc)| matches!(self.player.cells[*nr][*nc], Visible::Flag))
            .count();
        if flags != usize::from(number) {
            return Err(ApplyError::ChordFlagCount {
                at: pos_of(r, c),
                flags: flags as u32,
                number,
            });
        }
        let hidden: Vec<(usize, usize)> = neighbors
            .into_iter()
            .filter(|(nr, nc)| matches!(self.player.cells[*nr][*nc], Visible::Hidden))
            .collect();
        if hidden.is_empty() {
            return Err(ApplyError::NothingToOpen { at: pos_of(r, c) });
        }
        for (nr, nc) in hidden {
            if self.player.status != Status::Playing {
                break;
            }
            if matches!(self.player.cells[nr][nc], Visible::Hidden) {
                self.reveal(nr, nc)?;
            }
        }
        Ok(())
    }

    fn reveal(&mut self, r: usize, c: usize) -> Result<(), ApplyError> {
        match self.truth.cells[r][c] {
            -1 => {
                self.player.cells[r][c] = Visible::Exploded;
                self.player.status = Status::Lost;
                self.player.exploded = Some(pos_of(r, c));
                Ok(())
            }
            0 => {
                self.flood(r, c);
                self.mark_won_if_clear();
                Ok(())
            }
            n @ 1..=8 => {
                self.player.cells[r][c] = Visible::Number(n as u8);
                self.mark_won_if_clear();
                Ok(())
            }
            other => Err(ApplyError::CorruptTruth { value: other }),
        }
    }

    fn flood(&mut self, start_r: usize, start_c: usize) {
        let rows = self.player.rows as usize;
        let cols = self.player.cols as usize;
        let mut stack = vec![(start_r, start_c)];
        while let Some((r, c)) = stack.pop() {
            if !matches!(self.player.cells[r][c], Visible::Hidden) {
                continue;
            }
            let number = self.truth.cells[r][c];
            if number < 0 {
                continue;
            }
            self.player.cells[r][c] = Visible::Number(number as u8);
            if number == 0 {
                stack.extend(neighbor_indexes(rows, cols, r, c));
            }
        }
    }

    fn mark_won_if_clear(&mut self) {
        let clear = self
            .truth
            .cells
            .iter()
            .flatten()
            .zip(self.player.cells.iter().flatten())
            .all(|(truth, visible)| *truth < 0 || matches!(visible, Visible::Number(_)));
        if clear {
            self.player.status = Status::Won;
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GameError {
    EmptyBoard,
    MineOutOfRange { row: u32, col: u32 },
    DuplicateMine { row: u32, col: u32 },
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameError::EmptyBoard => write!(f, "board size must be at least 1x1"),
            GameError::MineOutOfRange { row, col } => {
                write!(f, "mine at row={row} col={col} is outside the board")
            }
            GameError::DuplicateMine { row, col } => {
                write!(f, "duplicate mine at row={row} col={col}")
            }
        }
    }
}

impl std::error::Error for GameError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyError {
    Terminal,
    OutOfRange { row: u32, col: u32 },
    AlreadyOpen { at: CellPos },
    Flagged { at: CellPos },
    NotANumber { at: CellPos },
    ChordFlagCount { at: CellPos, flags: u32, number: u8 },
    NothingToOpen { at: CellPos },
    CorruptTruth { value: i8 },
}

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApplyError::Terminal => write!(f, "the game is already over"),
            ApplyError::OutOfRange { row, col } => {
                write!(f, "row={row} col={col} is outside the board")
            }
            ApplyError::AlreadyOpen { at } => write!(f, "{at} is already open"),
            ApplyError::Flagged { at } => write!(f, "{at} is flagged"),
            ApplyError::NotANumber { at } => write!(f, "{at} is not an opened number"),
            ApplyError::ChordFlagCount { at, flags, number } => {
                write!(f, "chord at {at} has {flags} flags around a {number}")
            }
            ApplyError::NothingToOpen { at } => write!(f, "chord at {at} opens nothing"),
            ApplyError::CorruptTruth { value } => write!(f, "hidden cell value {value} is invalid"),
        }
    }
}

impl std::error::Error for ApplyError {}

pub(crate) fn index_of(
    rows: u32,
    cols: u32,
    row: u32,
    col: u32,
) -> Result<(usize, usize), ApplyError> {
    if row == 0 || col == 0 || row > rows || col > cols {
        return Err(ApplyError::OutOfRange { row, col });
    }
    Ok(((row - 1) as usize, (col - 1) as usize))
}

fn pos_of(r: usize, c: usize) -> CellPos {
    CellPos::new(r as u32 + 1, c as u32 + 1)
}

pub(crate) fn neighbor_indexes(
    rows: usize,
    cols: usize,
    r: usize,
    c: usize,
) -> Vec<(usize, usize)> {
    let mut cells = Vec::with_capacity(8);
    for dr in -1..=1 {
        for dc in -1..=1 {
            if dr == 0 && dc == 0 {
                continue;
            }
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                cells.push((nr as usize, nc as usize));
            }
        }
    }
    cells
}

fn number_grid(rows: usize, cols: usize, mines: &HashSet<(usize, usize)>) -> Vec<Vec<i8>> {
    let mut cells = vec![vec![0i8; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            if mines.contains(&(r, c)) {
                cells[r][c] = -1;
                continue;
            }
            let count = neighbor_indexes(rows, cols, r, c)
                .into_iter()
                .filter(|cell| mines.contains(cell))
                .count();
            cells[r][c] = count as i8;
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corner() -> Game {
        Game::from_mines(3, 3, &[CellPos::new(1, 1), CellPos::new(3, 3)]).unwrap()
    }

    #[test]
    fn opening_a_zero_matches_the_textbook_view() {
        let mut game = corner();
        game.apply(Action::open(1, 3)).unwrap();
        assert_eq!(game.player().get(1, 1), Some(Visible::Hidden));
        assert_eq!(game.player().get(1, 2), Some(Visible::Number(1)));
        assert_eq!(game.player().get(1, 3), Some(Visible::Number(0)));
        assert_eq!(game.player().get(2, 1), Some(Visible::Hidden));
        assert_eq!(game.player().get(2, 2), Some(Visible::Number(2)));
        assert_eq!(game.player().get(2, 3), Some(Visible::Number(1)));
        assert_eq!(game.player().status(), Status::Playing);
        assert_eq!(game.truth().bbbv() > 0, true);
    }

    #[test]
    fn opening_a_mine_ends_the_game_without_revealing_the_rest() {
        let mut game = corner();
        game.apply(Action::open(1, 1)).unwrap();
        assert_eq!(game.player().status(), Status::Lost);
        assert_eq!(game.player().get(1, 1), Some(Visible::Exploded));
        assert_eq!(game.player().get(3, 3), Some(Visible::Hidden));
        assert!(game.apply(Action::open(1, 3)).is_err());
    }

    #[test]
    fn flag_toggles_and_blocks_open() {
        let mut game = corner();
        game.apply(Action::flag(1, 1)).unwrap();
        assert_eq!(game.player().get(1, 1), Some(Visible::Flag));
        assert_eq!(game.player().flags_left(), 1);
        assert!(matches!(
            game.apply(Action::open(1, 1)),
            Err(ApplyError::Flagged { .. })
        ));
        game.apply(Action::flag(1, 1)).unwrap();
        assert_eq!(game.player().get(1, 1), Some(Visible::Hidden));
    }
}

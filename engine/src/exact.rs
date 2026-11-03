//! Mine layouts consistent with the opened numbers and `mines_total`.
//!
//! Flags are marks, so hidden and flagged cells are both unknown. Frontier
//! cells touch an opened number; every other unknown cell is interior. Each
//! frontier component is searched on its own. A component solution whose mine
//! count is `k` is kept only when the mines still to place, `mines_total - k`
//! over every component, fit in the interior. A cell is safe when it is clear
//! in every kept layout, and a mine when it is a mine in every kept layout.
//! How many layouts agree is not used: one kept layout on each side is enough
//! to leave the cell undecided.
//!
//! [`forced`](crate::forced) still stops at the first tier that finds any
//! cell, then applies only the two global completions. This pass does not
//! stop early, so a cell that is safe only because of the mine total is
//! reported even when some other cell was already forced by a shallower rule.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::game::{index_of, neighbor_indexes, ActionKind, CellPos, PlayerBoard, Status, Visible};
use crate::oracle::{
    run_direct, run_enumerate, run_subset, solver_board, ForceError, Tier, MAX_ENUM_CELLS,
};

/// One supported action, with the shallowest tier that proves the cells it uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExactAction {
    pub kind: ActionKind,
    pub at: CellPos,
    pub tier: Tier,
}

/// Safe cells, mine cells, and the actions those conclusions support.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exact {
    pub safe: BTreeSet<CellPos>,
    pub mines: BTreeSet<CellPos>,
    pub actions: Vec<ExactAction>,
    pub cells: BTreeMap<CellPos, Tier>,
}

/// Cells decided by every mine layout that matches the visible board.
pub fn exact(player: &PlayerBoard) -> Result<Exact, ForceError> {
    if player.status() != Status::Playing {
        return Err(ForceError::Terminal);
    }
    let (safe, mines) = decided_cells(player)?;
    let local = local_tiers(player)?;
    let mut cells = BTreeMap::new();
    for cell in safe.iter().chain(mines.iter()) {
        let tier = local.get(cell).copied().unwrap_or(Tier::Global);
        cells.insert(*cell, tier);
    }
    let actions = supported_actions(player, &safe, &mines, &cells);
    Ok(Exact {
        safe,
        mines,
        actions,
        cells,
    })
}

/// Shallowest local tier that proves each cell, tried independently.
///
/// Single, subset, and enumeration each see the original board. A hit in an
/// earlier tier does not skip the later ones, and single-cell writeback stays
/// inside the single pass. Enumeration wider than the cap is skipped here;
/// a cell that pass cannot reach is Global when the mine total still proves it.
fn local_tiers(player: &PlayerBoard) -> Result<BTreeMap<CellPos, Tier>, ForceError> {
    let board = solver_board(player);
    let mut tiers = BTreeMap::new();
    remember(&mut tiers, run_direct(&board)?, Tier::Single);
    remember(&mut tiers, run_subset(&board)?, Tier::Subset);
    match run_enumerate(&board) {
        Ok(found) => remember(&mut tiers, found, Tier::Enumerate),
        Err(ForceError::EnumLimit { .. }) => {}
        Err(error) => return Err(error),
    }
    Ok(tiers)
}

fn remember(
    tiers: &mut BTreeMap<CellPos, Tier>,
    found: (BTreeSet<CellPos>, BTreeSet<CellPos>),
    tier: Tier,
) {
    for cell in found.0.into_iter().chain(found.1) {
        tiers.entry(cell).or_insert(tier);
    }
}

fn decided_cells(
    player: &PlayerBoard,
) -> Result<(BTreeSet<CellPos>, BTreeSet<CellPos>), ForceError> {
    let unknown = unknown_cells(player);
    let frontier: Vec<CellPos> = unknown
        .iter()
        .copied()
        .filter(|cell| touches_number(player, *cell))
        .collect();
    let interior: Vec<CellPos> = unknown
        .iter()
        .copied()
        .filter(|cell| !touches_number(player, *cell))
        .collect();
    let constraints = frontier_constraints(player, &frontier)?;
    let parts = components(&frontier, &constraints)?;
    let total = player.mines_total();
    let interior_n = interior.len() as u32;
    let feasible = feasible_totals(&parts, total, interior_n);
    if feasible.is_empty() {
        return Err(ForceError::Inconsistent);
    }

    let mut safe = BTreeSet::new();
    let mut mines = BTreeSet::new();
    for (index, part) in parts.iter().enumerate() {
        let rests = sums_except(&parts, index);
        for (local, cell) in part.cells.iter().enumerate() {
            let can_mine = part.mine_ks[local]
                .iter()
                .any(|k| fits(*k, &rests, total, interior_n));
            let can_safe = part.safe_ks[local]
                .iter()
                .any(|k| fits(*k, &rests, total, interior_n));
            match (can_mine, can_safe) {
                (true, false) => {
                    mines.insert(*cell);
                }
                (false, true) => {
                    safe.insert(*cell);
                }
                (true, true) => {}
                (false, false) => return Err(ForceError::Inconsistent),
            }
        }
    }

    let remainders: BTreeSet<u32> = feasible.into_iter().map(|k| total - k).collect();
    if let Some(only) = only_value(&remainders) {
        if only == 0 {
            safe.extend(interior);
        } else if only == interior_n {
            mines.extend(interior);
        }
    }
    Ok((safe, mines))
}

struct Constraint {
    cells: Vec<usize>,
    need: u8,
}

struct Part {
    cells: Vec<CellPos>,
    possible_k: BTreeSet<u32>,
    mine_ks: Vec<BTreeSet<u32>>,
    safe_ks: Vec<BTreeSet<u32>>,
}

fn frontier_constraints(
    player: &PlayerBoard,
    frontier: &[CellPos],
) -> Result<Vec<Constraint>, ForceError> {
    let index: BTreeMap<CellPos, usize> = frontier
        .iter()
        .enumerate()
        .map(|(at, cell)| (*cell, at))
        .collect();
    let mut constraints = Vec::new();
    for row in 1..=player.rows() {
        for col in 1..=player.cols() {
            let Visible::Number(number) = player.get(row, col).expect("in range") else {
                continue;
            };
            let (r, c) = index_of(player.rows(), player.cols(), row, col).expect("in range");
            let mut cells = Vec::new();
            for (nr, nc) in neighbor_indexes(player.rows() as usize, player.cols() as usize, r, c) {
                let pos = CellPos::new(nr as u32 + 1, nc as u32 + 1);
                if let Some(at) = index.get(&pos) {
                    cells.push(*at);
                }
            }
            if cells.is_empty() {
                if number != 0 {
                    return Err(ForceError::Inconsistent);
                }
                continue;
            }
            if usize::from(number) > cells.len() {
                return Err(ForceError::Inconsistent);
            }
            constraints.push(Constraint {
                cells,
                need: number,
            });
        }
    }
    Ok(constraints)
}

fn components(frontier: &[CellPos], constraints: &[Constraint]) -> Result<Vec<Part>, ForceError> {
    let mut parent: Vec<usize> = (0..frontier.len()).collect();
    for constraint in constraints {
        if let Some(first) = constraint.cells.first().copied() {
            for other in constraint.cells.iter().skip(1).copied() {
                union(&mut parent, first, other);
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for cell in 0..frontier.len() {
        groups
            .entry(find(&mut parent, cell))
            .or_default()
            .push(cell);
    }

    let mut parts = Vec::new();
    for group in groups.into_values() {
        if group.len() > MAX_ENUM_CELLS {
            return Err(ForceError::EnumLimit { cells: group.len() });
        }
        let mut local_of = vec![None; frontier.len()];
        for (local, global) in group.iter().copied().enumerate() {
            local_of[global] = Some(local);
        }
        let local_constraints: Vec<Constraint> = constraints
            .iter()
            .filter_map(|constraint| {
                let cells: Vec<usize> = constraint
                    .cells
                    .iter()
                    .filter_map(|global| local_of[*global])
                    .collect();
                if cells.is_empty() {
                    None
                } else {
                    Some(Constraint {
                        cells,
                        need: constraint.need,
                    })
                }
            })
            .collect();
        let n = group.len();
        let mut possible_k = BTreeSet::new();
        let mut mine_ks = vec![BTreeSet::new(); n];
        let mut safe_ks = vec![BTreeSet::new(); n];
        let mut assign = vec![None; n];
        search(
            &local_constraints,
            &mut assign,
            &mut possible_k,
            &mut mine_ks,
            &mut safe_ks,
        );
        if possible_k.is_empty() {
            return Err(ForceError::Inconsistent);
        }
        parts.push(Part {
            cells: group.iter().map(|global| frontier[*global]).collect(),
            possible_k,
            mine_ks,
            safe_ks,
        });
    }
    Ok(parts)
}

fn search(
    constraints: &[Constraint],
    assign: &mut [Option<bool>],
    possible_k: &mut BTreeSet<u32>,
    mine_ks: &mut [BTreeSet<u32>],
    safe_ks: &mut [BTreeSet<u32>],
) {
    if !propagate(constraints, assign) {
        return;
    }
    let Some(index) = assign.iter().position(Option::is_none) else {
        let k = assign.iter().filter(|cell| **cell == Some(true)).count() as u32;
        possible_k.insert(k);
        for (at, cell) in assign.iter().enumerate() {
            if *cell == Some(true) {
                mine_ks[at].insert(k);
            } else {
                safe_ks[at].insert(k);
            }
        }
        return;
    };
    for value in [Some(false), Some(true)] {
        let mut next = assign.to_vec();
        next[index] = value;
        search(constraints, &mut next, possible_k, mine_ks, safe_ks);
    }
}

/// Fill cells that every remaining assignment of one constraint agrees on.
fn propagate(constraints: &[Constraint], assign: &mut [Option<bool>]) -> bool {
    let mut changed = true;
    while changed {
        changed = false;
        for constraint in constraints {
            let mut mines = 0u8;
            let mut free = Vec::new();
            for cell in constraint.cells.iter().copied() {
                match assign[cell] {
                    Some(true) => mines += 1,
                    Some(false) => {}
                    None => free.push(cell),
                }
            }
            if mines > constraint.need
                || usize::from(mines) + free.len() < usize::from(constraint.need)
            {
                return false;
            }
            if free.is_empty() {
                continue;
            }
            if mines == constraint.need {
                for cell in free {
                    assign[cell] = Some(false);
                }
                changed = true;
            } else if usize::from(mines) + free.len() == usize::from(constraint.need) {
                for cell in free {
                    assign[cell] = Some(true);
                }
                changed = true;
            }
        }
    }
    true
}

fn feasible_totals(parts: &[Part], total: u32, interior: u32) -> BTreeSet<u32> {
    sums_of(parts.iter())
        .into_iter()
        .filter(|k| *k <= total && total - k <= interior)
        .collect()
}

fn sums_except(parts: &[Part], skip: usize) -> BTreeSet<u32> {
    sums_of(
        parts
            .iter()
            .enumerate()
            .filter_map(|(index, part)| (index != skip).then_some(part)),
    )
}

fn sums_of<'a>(parts: impl Iterator<Item = &'a Part>) -> BTreeSet<u32> {
    let mut sums = BTreeSet::from([0]);
    for part in parts {
        let mut next = BTreeSet::new();
        for sum in &sums {
            for k in &part.possible_k {
                next.insert(sum + k);
            }
        }
        sums = next;
    }
    sums
}

fn fits(k: u32, rests: &BTreeSet<u32>, total: u32, interior: u32) -> bool {
    rests
        .iter()
        .any(|rest| k + rest <= total && total - (k + rest) <= interior)
}

fn only_value(values: &BTreeSet<u32>) -> Option<u32> {
    if values.len() == 1 {
        values.iter().next().copied()
    } else {
        None
    }
}

fn find(parent: &mut [usize], mut cell: usize) -> usize {
    while parent[cell] != cell {
        parent[cell] = parent[parent[cell]];
        cell = parent[cell];
    }
    cell
}

fn union(parent: &mut [usize], left: usize, right: usize) {
    let left = find(parent, left);
    let right = find(parent, right);
    if left != right {
        parent[right] = left;
    }
}

fn unknown_cells(player: &PlayerBoard) -> Vec<CellPos> {
    let mut cells = Vec::new();
    for row in 1..=player.rows() {
        for col in 1..=player.cols() {
            if matches!(
                player.get(row, col).expect("in range"),
                Visible::Hidden | Visible::Flag
            ) {
                cells.push(CellPos::new(row, col));
            }
        }
    }
    cells
}

fn touches_number(player: &PlayerBoard, cell: CellPos) -> bool {
    let (r, c) = index_of(player.rows(), player.cols(), cell.row, cell.col).expect("on board");
    neighbor_indexes(player.rows() as usize, player.cols() as usize, r, c)
        .into_iter()
        .any(|(nr, nc)| {
            matches!(
                player.get(nr as u32 + 1, nc as u32 + 1),
                Some(Visible::Number(_))
            )
        })
}

/// Opens of hidden safe cells, flags of hidden mines, and chords whose
/// hidden neighbors are all safe. A flag already on the board is not undone:
/// removing a mark is not a supported action.
fn supported_actions(
    player: &PlayerBoard,
    safe: &BTreeSet<CellPos>,
    mines: &BTreeSet<CellPos>,
    cells: &BTreeMap<CellPos, Tier>,
) -> Vec<ExactAction> {
    let mut actions = Vec::new();
    for cell in safe {
        if matches!(player.get(cell.row, cell.col), Some(Visible::Hidden)) {
            actions.push(scored(ActionKind::Open, *cell, cells));
        }
    }
    for cell in mines {
        if matches!(player.get(cell.row, cell.col), Some(Visible::Hidden)) {
            actions.push(scored(ActionKind::Flag, *cell, cells));
        }
    }
    for row in 1..=player.rows() {
        for col in 1..=player.cols() {
            let Some(tier) = chord_tier(player, safe, cells, row, col) else {
                continue;
            };
            actions.push(ExactAction {
                kind: ActionKind::Chord,
                at: CellPos::new(row, col),
                tier,
            });
        }
    }
    actions.sort_by_key(|action| (action.kind, action.at));
    actions
}

fn scored(kind: ActionKind, at: CellPos, cells: &BTreeMap<CellPos, Tier>) -> ExactAction {
    ExactAction {
        kind,
        at,
        tier: cells[&at],
    }
}

/// `Some` when a chord here opens at least one cell and every one of them is safe.
fn chord_tier(
    player: &PlayerBoard,
    safe: &BTreeSet<CellPos>,
    cells: &BTreeMap<CellPos, Tier>,
    row: u32,
    col: u32,
) -> Option<Tier> {
    let Visible::Number(number) = player.get(row, col).expect("in range") else {
        return None;
    };
    let (r, c) = index_of(player.rows(), player.cols(), row, col).expect("in range");
    let mut flags = 0usize;
    let mut hidden = Vec::new();
    for (nr, nc) in neighbor_indexes(player.rows() as usize, player.cols() as usize, r, c) {
        let at = CellPos::new(nr as u32 + 1, nc as u32 + 1);
        match player.get(at.row, at.col).expect("in range") {
            Visible::Flag => flags += 1,
            Visible::Hidden => hidden.push(at),
            Visible::Number(_) | Visible::Exploded => {}
        }
    }
    if flags != usize::from(number) || hidden.is_empty() {
        return None;
    }
    if hidden.iter().any(|cell| !safe.contains(cell)) {
        return None;
    }
    hidden.into_iter().map(|cell| cells[&cell]).max()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::game::{Action, Game, Visible};
    use crate::generate::{generate, GenerateRequest};
    use crate::oracle::{forced, Tier};

    #[test]
    fn textbook_corner_is_an_enumerate_open() {
        let mut game = Game::from_mines(3, 3, &[CellPos::new(1, 1), CellPos::new(3, 3)]).unwrap();
        game.apply(Action::open(1, 3)).unwrap();
        let judged = exact(game.player()).unwrap();
        let open = judged
            .actions
            .iter()
            .find(|action| action.kind == ActionKind::Open && action.at == CellPos::new(3, 1))
            .expect("open (3,1)");
        assert_eq!(open.tier, Tier::Enumerate);
        assert_same(game.player());
    }

    #[test]
    fn mine_count_completion_keeps_a_global_tier_on_the_distant_cell() {
        // Row 4 touches the opened zeros, so those cells are single-cell safe.
        // Row 5 touches no opened number. forced() still opens it, because the
        // single tier already proved the only mine, but that cell's own tier
        // is global.
        let mut board = visible_board(&["#1..", "11..", "....", "####", "####"]);
        board.set_mines_total(1);
        let early = forced(&board).unwrap();
        assert_eq!(early.tier, Some(Tier::Single));
        assert!(early.actions.contains(&Action::open(5, 1)));
        let judged = exact(&board).unwrap();
        assert_eq!(judged.cells.get(&CellPos::new(1, 1)), Some(&Tier::Single));
        assert_eq!(judged.cells.get(&CellPos::new(5, 1)), Some(&Tier::Global));
        assert_same(&board);
    }

    #[test]
    fn a_proven_safe_flag_is_not_an_unflag_action() {
        let mut board = visible_board(&["#1..", "11..", "....", "...F"]);
        board.set_mines_total(1);
        let judged = exact(&board).unwrap();
        assert!(judged.safe.contains(&CellPos::new(4, 4)));
        assert!(!judged
            .actions
            .iter()
            .any(|action| { action.at == CellPos::new(4, 4) }));
    }

    #[test]
    fn single_hit_does_not_hide_a_cell_that_only_the_mine_total_proves() {
        // (1,1) is a single-cell mine, so forced() stops in Single and never
        // counts the remaining mine in the {(1,6),(2,6)} pair. Row 5 touches
        // no opened number, and every kept layout leaves it zero mines.
        let mut board = visible_board(&["#1..1#", "11..1#", "....##", "######", "######"]);
        board.set_mines_total(2);
        let early = forced(&board).unwrap();
        assert_eq!(early.tier, Some(Tier::Single));
        assert!(early.actions.contains(&Action::flag(1, 1)));
        assert!(!early.actions.contains(&Action::open(5, 1)));

        let judged = exact(&board).unwrap();
        assert!(judged.safe.contains(&CellPos::new(5, 1)));
        assert!(judged.safe.contains(&CellPos::new(4, 6)));
        let open = judged
            .actions
            .iter()
            .find(|action| action.kind == ActionKind::Open && action.at == CellPos::new(5, 1))
            .expect("open (5,1)");
        assert_eq!(open.tier, Tier::Global);
        assert_eq!(judged.cells.get(&CellPos::new(1, 1)), Some(&Tier::Single));
        assert_same(&board);
    }

    #[test]
    fn a_number_with_no_room_for_its_mines_is_inconsistent() {
        let mut board = visible_board(&["1"]);
        board.set_mines_total(0);
        assert!(matches!(exact(&board), Err(ForceError::Inconsistent)));
    }

    #[test]
    fn a_frontier_wider_than_the_cap_is_rejected() {
        let cols = 30;
        let mut cells = Vec::with_capacity(3);
        cells.push(vec![Visible::Hidden; cols]);
        let mut middle = vec![Visible::Number(2); cols];
        middle[0] = Visible::Number(1);
        middle[cols - 1] = Visible::Number(1);
        cells.push(middle);
        cells.push(vec![Visible::Hidden; cols]);
        let board = PlayerBoard::from_visible(cells);
        match exact(&board) {
            Err(ForceError::EnumLimit { cells }) => assert!(cells > MAX_ENUM_CELLS),
            other => panic!("expected enum limit, got {other:?}"),
        }
    }

    #[test]
    fn exact_matches_brute_force_on_random_small_midgames() {
        let mut rng = StdRng::seed_from_u64(2026_11_03);
        let mut checked = 0u32;
        for _round in 0..520 {
            let rows = rng.gen_range(3..=6);
            let cols = rng.gen_range(3..=6);
            let mine_n = rng.gen_range(1..=4);
            let Some(mut game) = random_game(&mut rng, rows, cols, mine_n) else {
                continue;
            };
            let extra = rng.gen_range(0..=5);
            for _step in 0..extra {
                if game.player().status() != Status::Playing {
                    break;
                }
                let safe = hidden_safe(&game);
                if safe.is_empty() {
                    break;
                }
                let cell = safe[rng.gen_range(0..safe.len())];
                game.apply(Action::open(cell.row, cell.col)).unwrap();
            }
            if game.player().status() != Status::Playing {
                continue;
            }
            assert_same(game.player());
            checked += 1;
        }
        assert!(checked >= 300, "only compared {checked} small boards");
    }

    #[test]
    fn exact_matches_brute_force_on_beginner_midgames() {
        let mut rng = StdRng::seed_from_u64(2026_11_03);
        let mut checked = 0u32;
        for seed in 1..=30 {
            let request = GenerateRequest::new(seed, 9, 9, 10).attempts(80);
            let Ok(fixture) = generate(&request) else {
                continue;
            };
            let mut game = Game::from_mines(fixture.rows, fixture.cols, &fixture.mines).unwrap();
            game.apply(Action::open(
                fixture.first_click.row,
                fixture.first_click.col,
            ))
            .unwrap();
            shrink_unknown(&mut game, &mut rng, 16);
            checked += compare_while_small(&mut game, &mut rng, 4);
        }
        assert!(
            checked >= 20,
            "only compared {checked} beginner mid-states under the brute cap"
        );
    }

    fn compare_while_small(game: &mut Game, rng: &mut StdRng, extra: usize) -> u32 {
        let mut checked = 0u32;
        if game.player().status() == Status::Playing && brute_can_run(game.player()) {
            assert_same(game.player());
            checked += 1;
        }
        for _step in 0..extra {
            if game.player().status() != Status::Playing {
                break;
            }
            let safe = hidden_safe(game);
            if safe.is_empty() {
                break;
            }
            let cell = safe[rng.gen_range(0..safe.len())];
            game.apply(Action::open(cell.row, cell.col)).unwrap();
            if game.player().status() == Status::Playing && brute_can_run(game.player()) {
                assert_same(game.player());
                checked += 1;
            }
        }
        checked
    }

    fn shrink_unknown(game: &mut Game, rng: &mut StdRng, cap: usize) {
        while game.player().status() == Status::Playing && unknown_len(game.player()) > cap {
            let safe = hidden_safe(game);
            if safe.is_empty() {
                break;
            }
            let cell = safe[rng.gen_range(0..safe.len())];
            game.apply(Action::open(cell.row, cell.col)).unwrap();
        }
    }

    fn brute_can_run(player: &PlayerBoard) -> bool {
        let unknown = unknown_cells(player).len();
        let mines = player.mines_total() as usize;
        combinations(unknown, mines).is_some_and(|count| count <= 200_000)
    }

    fn assert_same(player: &PlayerBoard) {
        let judged = exact(player).expect("exact layout");
        let (safe, mines) = brute(player).expect("brute layout");
        assert_eq!(judged.safe, safe);
        assert_eq!(judged.mines, mines);
        let from_exact: BTreeSet<_> = judged
            .actions
            .iter()
            .map(|action| (action.kind, action.at))
            .collect();
        assert_eq!(from_exact, action_keys(player, &safe, &mines));
    }

    fn action_keys(
        player: &PlayerBoard,
        safe: &BTreeSet<CellPos>,
        mines: &BTreeSet<CellPos>,
    ) -> BTreeSet<(ActionKind, CellPos)> {
        // Tiers are not part of the brute comparison. Any tier satisfies the lookup.
        let mut cells = BTreeMap::new();
        for cell in safe.iter().chain(mines.iter()) {
            cells.insert(*cell, Tier::Global);
        }
        supported_actions(player, safe, mines, &cells)
            .into_iter()
            .map(|action| (action.kind, action.at))
            .collect()
    }

    fn brute(player: &PlayerBoard) -> Result<(BTreeSet<CellPos>, BTreeSet<CellPos>), ForceError> {
        if player.status() != Status::Playing {
            return Err(ForceError::Terminal);
        }
        let unknown = unknown_cells(player);
        let total = player.mines_total() as usize;
        if total > unknown.len() {
            return Err(ForceError::Inconsistent);
        }
        let mut choose = vec![false; unknown.len()];
        let mut mine_hits = vec![0u32; unknown.len()];
        let mut solutions = 0u32;
        choose_mines(
            player,
            &unknown,
            &mut choose,
            total,
            0,
            &mut mine_hits,
            &mut solutions,
        );
        if solutions == 0 {
            return Err(ForceError::Inconsistent);
        }
        let mut safe = BTreeSet::new();
        let mut mines = BTreeSet::new();
        for (at, cell) in unknown.iter().enumerate() {
            if mine_hits[at] == 0 {
                safe.insert(*cell);
            } else if mine_hits[at] == solutions {
                mines.insert(*cell);
            }
        }
        Ok((safe, mines))
    }

    fn choose_mines(
        player: &PlayerBoard,
        unknown: &[CellPos],
        choose: &mut [bool],
        left: usize,
        start: usize,
        mine_hits: &mut [u32],
        solutions: &mut u32,
    ) {
        if left == 0 {
            if layout_ok(player, unknown, choose) {
                *solutions += 1;
                for (at, placed) in choose.iter().enumerate() {
                    if *placed {
                        mine_hits[at] += 1;
                    }
                }
            }
            return;
        }
        for at in start..unknown.len() {
            if unknown.len() - at < left {
                break;
            }
            choose[at] = true;
            choose_mines(
                player,
                unknown,
                choose,
                left - 1,
                at + 1,
                mine_hits,
                solutions,
            );
            choose[at] = false;
        }
    }

    fn layout_ok(player: &PlayerBoard, unknown: &[CellPos], choose: &[bool]) -> bool {
        let mut mine = vec![vec![false; player.cols() as usize]; player.rows() as usize];
        for (at, cell) in unknown.iter().enumerate() {
            if choose[at] {
                mine[(cell.row - 1) as usize][(cell.col - 1) as usize] = true;
            }
        }
        for row in 1..=player.rows() {
            for col in 1..=player.cols() {
                let Visible::Number(number) = player.get(row, col).expect("in range") else {
                    continue;
                };
                let (r, c) = index_of(player.rows(), player.cols(), row, col).expect("in range");
                let count = neighbor_indexes(player.rows() as usize, player.cols() as usize, r, c)
                    .into_iter()
                    .filter(|(nr, nc)| mine[*nr][*nc])
                    .count();
                if count != usize::from(number) {
                    return false;
                }
            }
        }
        true
    }

    fn random_game(rng: &mut StdRng, rows: u32, cols: u32, mine_n: usize) -> Option<Game> {
        let mut cells = Vec::new();
        for row in 1..=rows {
            for col in 1..=cols {
                cells.push(CellPos::new(row, col));
            }
        }
        if mine_n >= cells.len() {
            return None;
        }
        for index in 0..cells.len() {
            let swap = rng.gen_range(index..cells.len());
            cells.swap(index, swap);
        }
        let mines = &cells[..mine_n];
        let game = Game::from_mines(rows, cols, mines).unwrap();
        let safe: Vec<CellPos> = cells[mine_n..].to_vec();
        if safe.is_empty() {
            return None;
        }
        let click = safe[rng.gen_range(0..safe.len())];
        let mut game = game;
        game.apply(Action::open(click.row, click.col)).unwrap();
        (game.player().status() == Status::Playing).then_some(game)
    }

    fn hidden_safe(game: &Game) -> Vec<CellPos> {
        let mut cells = Vec::new();
        for row in 1..=game.player().rows() {
            for col in 1..=game.player().cols() {
                let hidden = matches!(game.player().get(row, col), Some(Visible::Hidden));
                let safe = game.truth().get(row, col).is_some_and(|value| value >= 0);
                if hidden && safe {
                    cells.push(CellPos::new(row, col));
                }
            }
        }
        cells
    }

    fn unknown_len(player: &PlayerBoard) -> usize {
        unknown_cells(player).len()
    }

    fn combinations(n: usize, k: usize) -> Option<u64> {
        if k > n {
            return Some(0);
        }
        let k = k.min(n - k);
        let mut total = 1u64;
        for i in 0..k {
            total = total.checked_mul((n - i) as u64)? / (i as u64 + 1);
        }
        Some(total)
    }

    fn visible_board(rows: &[&str]) -> PlayerBoard {
        let cells = rows
            .iter()
            .map(|row| {
                row.chars()
                    .map(|glyph| match glyph {
                        '#' => Visible::Hidden,
                        'F' => Visible::Flag,
                        '.' => Visible::Number(0),
                        '1'..='8' => Visible::Number(glyph.to_digit(10).unwrap() as u8),
                        other => panic!("bad glyph {other}"),
                    })
                    .collect()
            })
            .collect();
        PlayerBoard::from_visible(cells)
    }
}

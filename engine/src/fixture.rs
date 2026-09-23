//! A fixture is one frozen board plus the open-only path that clears it.
//!
//! The first click is given. It is not a reasoning step. Each gold step was
//! forced from the board the player could see at that moment.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::game::{Action, ApplyError, CellPos, Game, Status};
use crate::oracle::{forced, ForceError, Tier, ORACLE_VERSION};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fixture {
    pub oracle_version: String,
    pub seed: u64,
    pub attempts: u32,
    pub rows: u32,
    pub cols: u32,
    pub mines_total: u32,
    pub first_click: CellPos,
    pub mines: Vec<CellPos>,
    pub bbbv: u32,
    pub max_tier: Tier,
    pub initial_text: String,
    pub gold: Vec<GoldStep>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldStep {
    pub action: Action,
    pub tier: Tier,
    pub text_after: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayError {
    Version {
        found: String,
    },
    BadShape(&'static str),
    Bbbv {
        found: u32,
        expected: u32,
    },
    Opening(ApplyError),
    InitialText,
    Forced {
        step: usize,
        error: ForceError,
    },
    Unsupported {
        step: usize,
        action: Action,
    },
    Tier {
        step: usize,
        found: Tier,
        expected: Tier,
    },
    Text {
        step: usize,
    },
    Apply {
        step: usize,
        error: ApplyError,
    },
    NotWon {
        status: Status,
    },
    MaxTier {
        found: Tier,
        expected: Tier,
    },
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReplayError::Version { found } => {
                write!(f, "oracle version {found} does not match {ORACLE_VERSION}")
            }
            ReplayError::BadShape(reason) => write!(f, "{reason}"),
            ReplayError::Bbbv { found, expected } => {
                write!(f, "3BV is {found}, fixture says {expected}")
            }
            ReplayError::Opening(error) => write!(f, "first click failed: {error}"),
            ReplayError::InitialText => write!(f, "opening board does not match the fixture"),
            ReplayError::Forced { step, error } => {
                write!(f, "step {step} could not be judged: {error}")
            }
            ReplayError::Unsupported { step, action } => {
                write!(f, "step {step} {action} is not forced")
            }
            ReplayError::Tier {
                step,
                found,
                expected,
            } => {
                write!(f, "step {step} is tier {found}, fixture says {expected}")
            }
            ReplayError::Text { step } => {
                write!(f, "board after step {step} does not match the fixture")
            }
            ReplayError::Apply { step, error } => {
                write!(f, "step {step} could not be played: {error}")
            }
            ReplayError::NotWon { status } => write!(f, "replay ended {status}"),
            ReplayError::MaxTier { found, expected } => {
                write!(f, "deepest tier is {found}, fixture says {expected}")
            }
        }
    }
}

impl std::error::Error for ReplayError {}

/// Rebuilds the hidden board and plays the gold path.
///
/// A step that is not in the oracle's forced set stops the replay.
pub fn replay(fixture: &Fixture) -> Result<(), ReplayError> {
    if fixture.oracle_version != ORACLE_VERSION {
        return Err(ReplayError::Version {
            found: fixture.oracle_version.clone(),
        });
    }
    if fixture.gold.is_empty() {
        return Err(ReplayError::BadShape("gold path is empty"));
    }
    if fixture.mines.len() != fixture.mines_total as usize {
        return Err(ReplayError::BadShape(
            "mine list length does not match mines_total",
        ));
    }
    let mut game = Game::from_mines(fixture.rows, fixture.cols, &fixture.mines)
        .map_err(|_| ReplayError::BadShape("mine list does not fit the board"))?;
    if game.truth().bbbv() != fixture.bbbv {
        return Err(ReplayError::Bbbv {
            found: game.truth().bbbv(),
            expected: fixture.bbbv,
        });
    }
    game.apply(Action::open(
        fixture.first_click.row,
        fixture.first_click.col,
    ))
    .map_err(ReplayError::Opening)?;
    if game.player().render() != fixture.initial_text {
        return Err(ReplayError::InitialText);
    }
    for (index, step) in fixture.gold.iter().enumerate() {
        let step_no = index + 1;
        let judged = forced(game.player()).map_err(|error| ReplayError::Forced {
            step: step_no,
            error,
        })?;
        let Some(found) = judged.tier else {
            return Err(ReplayError::Unsupported {
                step: step_no,
                action: step.action,
            });
        };
        if found != step.tier {
            return Err(ReplayError::Tier {
                step: step_no,
                found,
                expected: step.tier,
            });
        }
        if !judged.actions.contains(&step.action) {
            return Err(ReplayError::Unsupported {
                step: step_no,
                action: step.action,
            });
        }
        game.apply(step.action)
            .map_err(|error| ReplayError::Apply {
                step: step_no,
                error,
            })?;
        if game.player().render() != step.text_after {
            return Err(ReplayError::Text { step: step_no });
        }
    }
    if game.player().status() != Status::Won {
        return Err(ReplayError::NotWon {
            status: game.player().status(),
        });
    }
    let deepest = fixture
        .gold
        .iter()
        .map(|step| step.tier)
        .max()
        .expect("gold is non-empty");
    if deepest != fixture.max_tier {
        return Err(ReplayError::MaxTier {
            found: deepest,
            expected: fixture.max_tier,
        });
    }
    Ok(())
}

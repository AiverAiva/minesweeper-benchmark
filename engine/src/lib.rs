//! Minesweeper as two boards.
//!
//! [`Truth`] is the mine map. It is written into fixtures and used to apply
//! clicks. [`PlayerBoard`] is the only view a model is allowed to see.
//! [`forced`] decides legal deductions from that view.

mod fixture;
mod game;
mod generate;
mod oracle;
mod render;

pub use fixture::{replay, Fixture, GoldStep, ReplayError};
pub use game::{
    Action, ActionKind, ApplyError, CellPos, Game, GameError, PlayerBoard, Status, Truth, Visible,
};
pub use generate::{
    generate, play_forced, GenerateError, GenerateRequest, PlayError, DEFAULT_ATTEMPTS,
};
pub use oracle::{forced, ForceError, Forced, Tier, MAX_ENUM_CELLS, ORACLE_VERSION};

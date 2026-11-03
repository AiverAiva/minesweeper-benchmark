//! Minesweeper as two boards.
//!
//! [`Truth`] is the mine map. It is written into fixtures and used to apply
//! clicks. [`PlayerBoard`] is the only view a model is allowed to see.
//! [`forced`] decides the tiered deductions used to build a gold path.
//! [`exact`] decides which cells the mine total makes safe or mined; scoring
//! uses that, not whether a shallower tier already returned.

mod batch;
mod exact;
mod fixture;
mod game;
mod generate;
mod oracle;
mod render;
mod session;

pub use batch::{
    generate_batch, BatchError, BatchEvent, BatchReport, BatchRequest, Catalog, CatalogEntry,
    Overwrite, SeedOutcome, CATALOG_KIND, EXHAUSTED_KIND,
};
pub use exact::{exact, exact_limited, Exact, ExactAction, ExactStatus, DEFAULT_NODE_LIMIT};
pub use fixture::{replay, Fixture, GoldStep, ReplayError};
pub use game::{
    Action, ActionKind, ApplyError, CellPos, Game, GameError, PlayerBoard, Status, Truth, Visible,
};
pub use generate::{
    generate, play_forced, GenerateError, GenerateRequest, PlayError, DEFAULT_ATTEMPTS,
};
pub use oracle::{forced, ForceError, Forced, Tier, MAX_ENUM_CELLS, ORACLE_VERSION};
pub use session::{handle_session_line, run_session};

//! Line protocol between the text harness and the rules.
//!
//! One JSON object arrives on stdin. One JSON object leaves on stdout.
//! Piped stdout is block-buffered, so every response is flushed.

use std::io::{self, BufRead, Write};

use serde_json::{json, Value};

use crate::fixture::Fixture;
use crate::game::{Action, ApplyError, CellPos, Game};
use crate::oracle::{forced, ForceError, ORACLE_VERSION};

pub fn run_session() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut game = None;
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = handle_session_line(&mut game, &line);
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }
    Ok(())
}

pub fn handle_session_line(game: &mut Option<Game>, line: &str) -> Value {
    let value: Value = match serde_json::from_str(line) {
        Ok(value) => value,
        Err(error) => return fail(None, "parse", error.to_string(), None),
    };
    match value.get("op").and_then(Value::as_str) {
        Some("ping") => json!({"ok": true, "op": "pong"}),
        Some("new") => op_new(game, &value),
        Some("new_fixture") => op_new_fixture(game, &value),
        Some("forced") => op_forced(game),
        Some("apply") => op_apply(game, &value),
        Some(other) => fail(Some(other), "parse", "unknown op".to_string(), None),
        None => fail(None, "parse", "missing op".to_string(), None),
    }
}

fn op_new(slot: &mut Option<Game>, value: &Value) -> Value {
    let (rows, cols) = match board_size(value) {
        Ok(size) => size,
        Err(detail) => return fail(Some("new"), "parse", detail, None),
    };
    let mines = match cells_of(value.get("mines")) {
        Ok(mines) => mines,
        Err(detail) => return fail(Some("new"), "parse", detail, None),
    };
    let click = match cell_of(value.get("first_click")) {
        Ok(click) => click,
        Err(detail) => return fail(Some("new"), "parse", detail, None),
    };
    match load_opening(rows, cols, &mines, click) {
        Ok(game) => {
            let response = state_value("new", &game);
            *slot = Some(game);
            response
        }
        Err((error, detail)) => fail(Some("new"), error, detail, None),
    }
}

fn op_new_fixture(slot: &mut Option<Game>, value: &Value) -> Value {
    let Some(body) = value.get("fixture") else {
        return fail(
            Some("new_fixture"),
            "parse",
            "missing fixture".to_string(),
            None,
        );
    };
    let fixture: Fixture = match serde_json::from_value(body.clone()) {
        Ok(fixture) => fixture,
        Err(error) => return fail(Some("new_fixture"), "parse", error.to_string(), None),
    };
    if fixture.oracle_version != ORACLE_VERSION {
        return fail(
            Some("new_fixture"),
            "version",
            format!(
                "fixture oracle is {}, engine is {ORACLE_VERSION}",
                fixture.oracle_version
            ),
            None,
        );
    }
    if fixture.mines.len() as u32 != fixture.mines_total {
        return fail(
            Some("new_fixture"),
            "bad_game",
            "mine count does not match mines_total".to_string(),
            None,
        );
    }
    match load_opening(
        fixture.rows,
        fixture.cols,
        &fixture.mines,
        fixture.first_click,
    ) {
        Ok(game) => {
            let text = game.player().render();
            if text != fixture.initial_text {
                return fail(
                    Some("new_fixture"),
                    "initial_text",
                    "rendered opening does not match fixture.initial_text".to_string(),
                    None,
                );
            }
            let response = state_value("new_fixture", &game);
            *slot = Some(game);
            response
        }
        Err((error, detail)) => fail(Some("new_fixture"), error, detail, None),
    }
}

fn load_opening(
    rows: u32,
    cols: u32,
    mines: &[CellPos],
    click: CellPos,
) -> Result<Game, (&'static str, String)> {
    let mut game = match Game::from_mines(rows, cols, mines) {
        Ok(game) => game,
        Err(error) => return Err(("bad_game", error.to_string())),
    };
    if let Err(error) = game.apply(Action::open(click.row, click.col)) {
        return Err(("bad_click", error.to_string()));
    }
    Ok(game)
}

fn op_forced(slot: &mut Option<Game>) -> Value {
    let Some(game) = slot else {
        return fail(
            Some("forced"),
            "no_game",
            "no game is loaded".to_string(),
            None,
        );
    };
    match forced(game.player()) {
        Ok(judged) => json!({
            "ok": true,
            "op": "forced",
            "tier": judged.tier,
            "actions": judged.actions,
        }),
        Err(ForceError::Terminal) => fail(
            Some("forced"),
            "terminal",
            ForceError::Terminal.to_string(),
            None,
        ),
        Err(ForceError::EnumLimit { cells }) => fail(
            Some("forced"),
            "enum_limit",
            ForceError::EnumLimit { cells }.to_string(),
            Some(cells),
        ),
        Err(ForceError::Inconsistent) => fail(
            Some("forced"),
            "inconsistent",
            ForceError::Inconsistent.to_string(),
            None,
        ),
    }
}

fn op_apply(slot: &mut Option<Game>, value: &Value) -> Value {
    let Some(game) = slot else {
        return fail(
            Some("apply"),
            "no_game",
            "no game is loaded".to_string(),
            None,
        );
    };
    let Some(body) = value.get("action") else {
        return fail(Some("apply"), "parse", "missing action".to_string(), None);
    };
    let action: Action = match serde_json::from_value(body.clone()) {
        Ok(action) => action,
        Err(error) => return fail(Some("apply"), "parse", error.to_string(), None),
    };
    match game.apply(action) {
        Ok(()) => state_value("apply", game),
        Err(ApplyError::Terminal) => fail(
            Some("apply"),
            "terminal",
            ApplyError::Terminal.to_string(),
            None,
        ),
        Err(error) => fail(Some("apply"), "illegal", error.to_string(), None),
    }
}

fn board_size(value: &Value) -> Result<(u32, u32), String> {
    let rows = required_u32(value, "rows")?;
    let cols = required_u32(value, "cols")?;
    Ok((rows, cols))
}

fn required_u32(value: &Value, key: &str) -> Result<u32, String> {
    match value.get(key).and_then(Value::as_u64) {
        Some(number) if number <= u64::from(u32::MAX) => Ok(number as u32),
        _ => Err(format!("{key} must be an integer")),
    }
}

fn cells_of(value: Option<&Value>) -> Result<Vec<CellPos>, String> {
    let Some(Value::Array(items)) = value else {
        return Err("mines must be a list of cells".to_string());
    };
    items.iter().map(|item| cell_of(Some(item))).collect()
}

fn cell_of(value: Option<&Value>) -> Result<CellPos, String> {
    let Some(value) = value else {
        return Err("missing cell".to_string());
    };
    Ok(CellPos::new(
        required_u32(value, "row")?,
        required_u32(value, "col")?,
    ))
}

fn state_value(op: &str, game: &Game) -> Value {
    json!({
        "ok": true,
        "op": op,
        "status": game.player().status(),
        "text": game.player().render(),
    })
}

fn fail(op: Option<&str>, error: &str, detail: String, cells: Option<usize>) -> Value {
    let mut body = json!({
        "ok": false,
        "error": error,
        "detail": detail,
    });
    if let Some(op) = op {
        body["op"] = json!(op);
    }
    if let Some(cells) = cells {
        body["cells"] = json!(cells);
    }
    body
}

#[cfg(test)]
mod tests {
    use super::handle_session_line;
    use crate::game::Status;
    use crate::oracle::ORACLE_VERSION;

    const NEW_TEXTBOOK: &str = r#"{"op":"new","rows":3,"cols":3,"mines":[{"row":1,"col":1},{"row":3,"col":3}],"first_click":{"row":1,"col":3}}"#;

    #[test]
    fn textbook_opening_forces_one_enumerate_open_and_wins() {
        let mut game = None;
        let created = handle_session_line(&mut game, NEW_TEXTBOOK);
        assert_eq!(created["ok"].as_bool(), Some(true));
        assert_eq!(created["status"].as_str(), Some("playing"));
        assert!(created["text"].as_str().unwrap().contains("R01  #1."));

        let judged = handle_session_line(&mut game, r#"{"op":"forced"}"#);
        assert_eq!(judged["tier"].as_str(), Some("enumerate"));
        let actions = judged["actions"].as_array().unwrap();
        assert!(actions.iter().any(|action| {
            action["kind"].as_str() == Some("open")
                && action["at"]["row"].as_u64() == Some(3)
                && action["at"]["col"].as_u64() == Some(1)
        }));

        let applied = handle_session_line(
            &mut game,
            r#"{"op":"apply","action":{"kind":"open","at":{"row":3,"col":1}}}"#,
        );
        assert_eq!(applied["ok"].as_bool(), Some(true));
        assert_eq!(applied["status"].as_str(), Some("won"));

        let again = handle_session_line(
            &mut game,
            r#"{"op":"apply","action":{"kind":"open","at":{"row":3,"col":1}}}"#,
        );
        assert_eq!(again["error"].as_str(), Some("terminal"));
    }

    #[test]
    fn an_illegal_apply_keeps_the_forced_set() {
        let mut game = None;
        handle_session_line(&mut game, NEW_TEXTBOOK);
        let before = handle_session_line(&mut game, r#"{"op":"forced"}"#);
        let illegal = handle_session_line(
            &mut game,
            r#"{"op":"apply","action":{"kind":"open","at":{"row":1,"col":2}}}"#,
        );
        assert_eq!(illegal["error"].as_str(), Some("illegal"));
        let after = handle_session_line(&mut game, r#"{"op":"forced"}"#);
        assert_eq!(before["tier"], after["tier"]);
        assert_eq!(before["actions"], after["actions"]);
    }

    #[test]
    fn a_bad_fixture_does_not_replace_the_loaded_game() {
        let mut game = None;
        handle_session_line(&mut game, NEW_TEXTBOOK);
        let request = serde_json::json!({
            "op": "new_fixture",
            "fixture": {
                "oracle_version": ORACLE_VERSION,
                "seed": 0,
                "attempts": 1,
                "rows": 3,
                "cols": 3,
                "mines_total": 2,
                "first_click": {"row": 1, "col": 3},
                "mines": [{"row": 1, "col": 1}, {"row": 3, "col": 3}],
                "bbbv": 2,
                "max_tier": "enumerate",
                "initial_text": "not the opening",
                "gold": [],
            }
        });
        let rejected = handle_session_line(&mut game, &request.to_string());
        assert_eq!(rejected["error"].as_str(), Some("initial_text"));
        assert_eq!(game.as_ref().unwrap().player().status(), Status::Playing);
        let judged = handle_session_line(&mut game, r#"{"op":"forced"}"#);
        assert_eq!(judged["ok"].as_bool(), Some(true));
    }

    #[test]
    fn ping_garbage_and_a_missing_game_are_distinct() {
        let mut game = None;
        let pong = handle_session_line(&mut game, r#"{"op":"ping"}"#);
        assert_eq!(pong["op"].as_str(), Some("pong"));
        assert_eq!(
            handle_session_line(&mut game, "nope")["error"].as_str(),
            Some("parse")
        );
        assert_eq!(
            handle_session_line(&mut game, r#"{"op":"dance"}"#)["error"].as_str(),
            Some("parse")
        );
        assert_eq!(
            handle_session_line(&mut game, r#"{"op":"forced"}"#)["error"].as_str(),
            Some("no_game")
        );
    }
}

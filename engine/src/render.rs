use crate::game::{PlayerBoard, Visible};

impl PlayerBoard {
    /// Compact text board. This is the view a person and a model both read.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("status: {}\n", self.status()));
        out.push_str(&format!("mines_total: {}\n", self.mines_total()));
        out.push_str(&format!("flags_left: {}\n", self.flags_left()));
        out.push_str(&format!("rows: {}\n", self.rows()));
        out.push_str(&format!("cols: {}\n", self.cols()));
        out.push_str("board:\n");
        for row in 1..=self.rows() {
            out.push_str(&format!("R{row:02}  "));
            for col in 1..=self.cols() {
                out.push(glyph(self.get(row, col).expect("row and col are in range")));
            }
            out.push('\n');
        }
        out
    }
}

fn glyph(cell: Visible) -> char {
    match cell {
        Visible::Hidden => '#',
        Visible::Flag => 'F',
        Visible::Number(0) => '.',
        Visible::Number(n) => char::from(b'0' + n),
        Visible::Exploded => '*',
    }
}

impl crate::game::Truth {
    /// Answer key. `*` is a mine. This string is not a player view.
    pub fn render_answer(&self) -> String {
        let mut out = String::new();
        for row in 1..=self.rows() {
            for col in 1..=self.cols() {
                let cell = self.get(row, col).expect("row and col are in range");
                let glyph = if cell < 0 {
                    '*'
                } else if cell == 0 {
                    '.'
                } else {
                    char::from(b'0' + cell as u8)
                };
                out.push(glyph);
            }
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::{Action, CellPos, Game};

    #[test]
    fn textbook_opening_renders_as_compact_grid() {
        let mut game = Game::from_mines(3, 3, &[CellPos::new(1, 1), CellPos::new(3, 3)]).unwrap();
        game.apply(Action::open(1, 3)).unwrap();
        assert_eq!(
            game.player().render(),
            "\
status: playing
mines_total: 2
flags_left: 2
rows: 3
cols: 3
board:
R01  #1.
R02  #21
R03  ###
"
        );
        assert_eq!(game.truth().render_answer(), "*1.\n121\n.1*\n");
    }
}

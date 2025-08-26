use crate::IterState;

pub struct VisibleDrop(u8);

#[allow(unused)]
impl VisibleDrop {
    pub fn new(val: u8) -> Self {
        println!("new! id:{val}");
        Self(val)
    }

    pub fn get(&self) -> u8 {
        self.0
    }
}

impl Drop for VisibleDrop {
    fn drop(&mut self) {
        println!("drop! id:{}", self.0)
    }
}

impl IterState for VisibleDrop {
    type Value = u8;

    type Solution = u8;

    fn init_from_value(initial_point: Self::Value) -> Self {
        Self::new(initial_point)
    }

    fn to_sol(&self) -> Self::Solution {
        self.0
    }
}
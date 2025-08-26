//! error types

use std::time::Duration;

use thiserror::Error;

use crate::IterState;

#[derive(Debug, Error)]
#[error("Solver reach the max iteration: {max_iteration}.")]
pub struct ReachMaxIteration<S: IterState> {
    pub(crate) max_iteration: u64,
    pub(crate) final_state: S,
}

impl<S: IterState> ReachMaxIteration<S> {
    pub fn take_final_state(self) -> S {
        self.final_state
    }

    pub fn get_solution(&self) -> S::Solution {
        self.final_state.to_sol()
    }
}

#[derive(Debug, Error)]
#[error("Solver timeout: {}ms or {}s", timeout.as_millis(), timeout.as_secs_f64())]
pub struct TimeOut<S: IterState> {
    pub(crate) timeout: Duration,
    pub(crate) final_state: S
}

impl<S: IterState> TimeOut<S> {
    pub fn take_final_state(self) -> S {
        self.final_state
    }

    pub fn get_solution(&self) -> S::Solution {
        self.final_state.to_sol()
    }
}
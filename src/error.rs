//! Error types.

use std::time::Duration;

use thiserror::Error;

use crate::IterState;

/// Error indicating that the solver reached the maximum iteration limit before converging.
///
/// This error occurs when the solver exceeds the specified maximum number of iterations
/// without reaching termination condition. It contains both the iteration limit that was
/// reached and the final state the solver had achieved when the limit was exceeded.
#[derive(Clone, Debug, Error)]
#[error("Solver reached the maximum iteration: {max_iteration}.")]
pub struct ReachMaxIteration<S: IterState> {
    /// The maximum iteration count that was reached.
    pub max_iteration: u64,
    /// The final state of the solver at the time the iteration limit was reached.
    pub(crate) final_state: S,
}

impl<S: IterState> ReachMaxIteration<S> {
    /// Consumes self and returns the final state reached by the solver.
    pub fn take_final_state(self) -> S {
        self.final_state
    }

    /// Returns an immutable reference to the final state reached by the solver.
    pub fn final_state_ref(&self) -> &S {
        &self.final_state
    }

    /// Directly obtains [`IterState::Solution`] from the final state reached by the solver.
    pub fn get_solution(&self) -> S::Solution {
        self.final_state.to_sol()
    }
}

/// Error indicating that the solver timed out before completing.
///
/// This error occurs when the solver exceeds the specified time limit without
/// reaching a termination condition. It contains both the timeout duration and
/// the final state the solver had reached when the timeout occurred.
#[derive(Clone, Debug, Error)]
#[error("Solver timeout: {}ms or {}s", timeout.as_millis(), timeout.as_secs_f64())]
pub struct TimeOut<S: IterState> {
    /// The time limit that was reached.
    pub timeout: Duration,
    /// The final state of the solver at the time of timeout.
    pub(crate) final_state: S,
}

impl<S: IterState> TimeOut<S> {
    /// Consumes self and returns the final state reached by the solver.
    pub fn take_final_state(self) -> S {
        self.final_state
    }

    /// Returns an immutable reference to the final state reached by the solver.
    pub fn final_state_ref(&self) -> &S {
        &self.final_state
    }

    /// Directly obtains [`IterState::Solution`] from the final state reached by the solver.
    pub fn get_solution(&self) -> S::Solution {
        self.final_state.to_sol()
    }
}

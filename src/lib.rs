#![doc = include_str!("../README.md")]

use std::{marker::PhantomData, mem::{self, MaybeUninit}, time::{Duration, Instant}};

use crate::error::{ReachMaxIteration, TimeOut};

pub mod error;
mod utils;

/// Intermediate state of an iterative algorithm.
///
/// For many iterative algorithms, the intermediate state often requires a more complex structure to describe compared to the initial iteration value and the solution.
///
/// In practical iterative algorithms, it is often necessary to perform some simple computations or obtain algorithm metadata after obtaining the solution.
///
/// Therefore, IterState allows you to customize the intermediate state and separate the abstractions of value and solution.
///
/// If you expect the simplest behavior, this crate has already implemented `IterState` for basic data types `i*`, `u*`, and `f*`, where their associated types Value and Solution are themselves.
pub trait IterState: Sized {
    /// Type representing the value during iteration (e.g., intermediate computation results).
    type Value;

    /// Type representing the final solution.  
    type Solution;  

    /// Initializes the state from an initial value.  
    fn init_from_value(initial_point: Self::Value) -> Self;  

    /// Converts the current state into the solution.  
    fn to_sol(&self) -> Self::Solution;  
}

macro_rules! iterstate_impl {
    ($($ty:ty),*) => {
        $(
            impl IterState for $ty {
                type Value = $ty;
                type Solution = $ty;
                fn init_from_value(initial_point: Self::Value) -> Self {
                    initial_point
                }
                fn to_sol(&self) -> Self::Solution {
                    *self
                }
            }
        )*
    };
}

iterstate_impl!(
    u8,u16,u32,u64,u128,usize,
    i8,i16,i32,i64,i128,isize,
    f32, f64
);


/// Solver type.
/// 
/// Note: This type does not provide specific algorithms but allows you to customize iteration methods and stopping conditions.
/// 
/// The `Problem` generic parameter has no constraints, allowing you to define any kind of interface in the Problem generic for use by iteration methods and stopping conditions.
#[derive(Debug)]
pub struct Solver<State, Problem, IterFn, TermFn>
where
State: IterState,
IterFn: Fn(&State, &Problem) -> State,
TermFn: Fn(&State, &Problem) -> bool,
{
    /// Intermediate state of the solver (uninitialized at the start)
    state: MaybeUninit<State>,
    /// Placeholder for the problem type (no runtime storage)
    problem: PhantomData<Problem>,
    /// Function defining the iteration logic
    iter_fn: IterFn,
    /// Function defining the termination condition
    term_cond: TermFn,
}
impl<State, Problem, IterFn, TermFn> Solver<State, Problem, IterFn, TermFn>
where 
    State: IterState,
    IterFn: Fn(&State, &Problem) -> State,
    TermFn: Fn(&State, &Problem) -> bool
{
    /// Creates a Solver instance with the specified methods.
    ///
    /// Parameter `iter_fn` defines the state iteration rule for the Solver.
    ///
    /// Parameter `term_cond` specifies that the Solver stops iterating if and only if
    /// the condition is met: `term_cond` returns `true` in the current state.
    pub fn new(
        iterfn: IterFn,
        termcond: TermFn
    ) -> Self {
        Self { 
            state: MaybeUninit::uninit(), 
            problem: PhantomData::<Problem>, 
            iter_fn: iterfn, 
            term_cond: termcond 
        }
    }

    /// Solves the problem by executing iterative logic with the given initial value and specific problem.
    ///
    /// Note: If the algorithm defined by the solver contains logical errors, the solve function may enter an infinite loop.
    /// 
    /// Although this method receives `&mut self`, the same Solver can correctly call the solve method multiple times, with each call of the solve method not affecting the others.
    /// # Example
    /// ```no_run
    /// let mut solver = Solver::new(iter_fn, term_cond);
    /// 
    /// solver.solve(val1, prob1);
    /// 
    /// // A second solve can be performed here without causing logical errors
    /// solver.solve(val2, prob2);
    /// ```
    pub fn solve(&mut self, initial_point: State::Value, problem: &Problem) -> State::Solution {
        // init state
        let initial_state = State::init_from_value(initial_point);
        self.state.write(initial_state);
        
        // do iter
        loop {
            let state = unsafe { self.state.assume_init_mut() };

            *state = (self.iter_fn)(state, problem);

            // check termination cond
            if (self.term_cond)(state, problem) {
                break;
            }
        }
        
        let final_state = unsafe{ self.state.assume_init_ref() };

        let sol = final_state.to_sol();

        unsafe { self.state.assume_init_drop(); }

        sol
    }



    pub fn solve_with_max_iteration(
        &mut self,
        initial_point: State::Value,
        problem: &Problem,
        max_iteration: u64
    ) -> Result<State::Solution, error::ReachMaxIteration<State>> {
        let mut reach_max = true;

        // init state
        let initial_state = State::init_from_value(initial_point);
        self.state.write(initial_state);        
        
        for _iteration in 0..max_iteration {
            let state = unsafe { self.state.assume_init_mut() };

            *state = (self.iter_fn)(state, problem);

            // check termination cond
            if (self.term_cond)(state, problem) {
                reach_max = false;
                break;
            }
        }

        let ret_res = if !reach_max {
            let final_state = unsafe{ self.state.assume_init_ref() };
            let sol = final_state.to_sol();
            unsafe { self.state.assume_init_drop(); }
            Ok(sol)
        } else {
            let final_state = unsafe { self.state.assume_init_read() };
            Err(ReachMaxIteration{
                max_iteration: max_iteration, 
                final_state: final_state
            })
        };

        ret_res
    }

    pub fn solve_with_timeout(
        &mut self,
        initial_point: State::Value,
        problem: &Problem,
        timeout: Duration        
    ) -> Result<State::Solution, TimeOut<State>> {
        let start_time = Instant::now();
        let mut is_timeout = true;

        // init state
        let initial_state = State::init_from_value(initial_point);
        self.state.write(initial_state);
        
        // do iter
        loop {
            let state = unsafe { self.state.assume_init_mut() };

            *state = (self.iter_fn)(state, problem);

            // check termination cond
            if (self.term_cond)(state, problem) {
                is_timeout = false;
                break;
            }

            if start_time.elapsed() > timeout {
                break;
            }
        }

        if !is_timeout {
            let final_state = unsafe{ self.state.assume_init_ref() };

            let sol = final_state.to_sol();

            unsafe { self.state.assume_init_drop(); }

            Ok(sol)                
        } else {
            let final_state = unsafe { self.state.assume_init_read() };
            Err(TimeOut { timeout: timeout, final_state: final_state })            
        }

    }



    /// Consumes the `Solver` and generates an iterator that outputs the current State at each step based on the given initial value and problem.
    /// 
    /// Use this method when you want to manipulate the state at each step in detail.
    /// 
    /// # Example
    /// ```no_run
    /// let solver: Solver<f64, _, _, _> = Solver::new(iter_fn, term_cond);
    /// let mut iteration = 1usize;
    /// 
    /// for state_float in solver.into_iter(2.0, &problem) {
    ///     println!("the solution after {} iteration(s): {}", iteration, state_float);
    ///     iteration += 1;
    /// } 
    /// ```
    pub fn into_iter<'prob>(self, 
        initial_point: State::Value, 
        problem: &'prob Problem
    ) -> SolverIterater<'prob, State, Problem, IterFn, TermFn> {
        // init Slover
        let initial_state = State::init_from_value(initial_point);

        SolverIterater { 
            problem: problem,
            state: initial_state,
            iter_fn: self.iter_fn,
            term_cond: self.term_cond, 
            need_term: false
        }
    }

    /// Consumes `self` and returns a new `Solver` with the given new termination condition.
    pub fn change_term_cond<NewTermCond>(self, new_cond: NewTermCond) -> Solver<State, Problem, IterFn, NewTermCond> 
    where 
        NewTermCond: Fn(&State, &Problem) -> bool
    {
        Solver { 
            state: self.state, 
            problem: self.problem, 
            iter_fn: self.iter_fn,
            term_cond: new_cond 
        }
    }
}

impl<State, Problem, IterFn, TermFn> Clone for Solver<State, Problem, IterFn, TermFn>
where 
    State: IterState,
    IterFn: Fn(&State, &Problem) -> State + Clone,
    TermFn: Fn(&State, &Problem) -> bool + Clone
{
    fn clone(&self) -> Self {
        Self { state: MaybeUninit::uninit(), problem: self.problem.clone(), iter_fn: self.iter_fn.clone(), term_cond: self.term_cond.clone() }
    }
}

/// Solver iterator, used to step through the solving process and output the state of each iteration.
///
/// This struct is created by consuming (taking ownership) a configured `Solver`,
/// and implements the `Iterator` trait, allowing users to obtain intermediate solving states
/// step by step through loops. Each call to the `next` method applies one iteration function
/// and checks whether the termination condition is met.
/// 
/// # Examples
/// See the example for the [`Solver::into_iter`] method.
///
/// [`into_iter`]: crate::Solver::into_iter
#[derive(Debug)]
pub struct SolverIterater<'prob, State, Problem, IterFn, TermFn> 
where 
    State: IterState,
    IterFn: Fn(&State, &Problem) -> State,
    TermFn: Fn(&State, &Problem) -> bool
{
    state: State,
    iter_fn: IterFn,
    term_cond: TermFn,
    problem: &'prob Problem,
    need_term: bool
}

impl<'prob, State, Problem, IterFn, TermFn> Iterator for SolverIterater<'prob, State, Problem, IterFn, TermFn> 
where 
    State: IterState,
    IterFn: Fn(&State, &Problem) -> State,
    TermFn: Fn(&State, &Problem) -> bool
{
    type Item = State;

    fn next(&mut self) -> Option<Self::Item> {
        if self.need_term {
            return None;
        }
        if (self.term_cond)(&self.state, &self.problem) {
            self.need_term = true;
            let _state = (self.iter_fn)(&self.state, &self.problem);
            return Some(mem::replace(&mut self.state, _state));
        }

        let next_state = (self.iter_fn)(&self.state, &self.problem);
        
        let old_state = mem::replace(&mut self.state, next_state);
        Some(old_state) 
    }
}

#[cfg(test)]
mod test {
    mod newton {
        use crate::{IterState, Solver};

        fn f_and_df(x: f64) -> (f64, f64) {
            let fx = x.exp() - 1.5;
            let dfx = x.exp();
            (fx, dfx)            
        }
        #[test]
    fn show() {
        let iter_fn = |state: &f64, problem: &fn(f64) -> (f64, f64)| {
            let x_n = *state;
            let (fx, dfx) = problem(x_n);
            x_n - (fx / dfx)
        };

        let term_cond = |state: &f64, problem: &fn(f64) -> (f64, f64)| {
            let (fx, _) = problem(*state);
            fx.abs() < 1e-6
        };

        let mut solver = Solver::new(iter_fn, term_cond);

        let solution = solver.solve(1.5, &(f_and_df as fn(f64) -> (f64, f64)));

        println!("solver's solution: {}", solution);
        println!("use std function ln: {}", 1.5_f64.ln());
    }

        #[derive(Clone)]
        enum Equation {
            Exp {
                // $ae^x - k = 0$
                a: f64,
                k: f64
            },

            Square {
                // $ax^2 + bx + c = 0$ 
                a: f64,
                b: f64,
                c: f64
            }
        }

        impl Equation {
            fn calc(&self, val: f64) -> f64 {
                match self {
                    Self::Exp { a, k } => {
                        a * val.exp() - k
                    }

                    Self::Square { a, b, c } => {
                        let x2 = a * val * val;
                        let x1 = b * val;
                        let x0 = c;
                        x2 + x1 + x0
                    }
                }
            }

            fn diff(&self, val: f64) -> f64 {
                match self {
                    Self::Exp { a, k: _ } => {
                        a * val.exp() 
                    }

                    Self::Square { a, b, c: _ } => {
                        ((2. * a) * val) + b
                    }
                }                
            }
        }

        #[derive(Debug, Clone)]
        struct NewtonState(f64);

        impl IterState for NewtonState {
            type Value = f64;
        
            type Solution = f64;
        
            fn init_from_value(initial_point: Self::Value) -> Self {
                Self(initial_point)
            }
        
            fn to_sol(&self) -> Self::Solution {
                self.0
            }
        }

        #[test]
        fn test() {
            let iter_fn = |state: &NewtonState, problem: &Equation| {
                let x = state.0;
                let dx = problem.diff(x);
                let fx = problem.calc(x);

                let next_x = x - (fx / dx);

                NewtonState(next_x)
            };

            let term_cond = |state: &NewtonState, problem: &Equation| {
                let epsilon = 1e-6;
                problem.calc(state.0) < epsilon
            };

            let mut solver = Solver::new(iter_fn, term_cond);

            let mut solver1 = solver.clone().change_term_cond(|state, equation| {
                equation.calc(state.0) < 1e-9
            });

            let prob1 = (Equation::Exp { a: 2., k: 3. }, 2.);

            let cloned_and_change_cond_sol = solver1.solve(prob1.1, &prob1.0.clone());

            let prob2 = (Equation::Square { a: 2., b: -5., c: 3. }, 6.);

            let prob1_sol = solver.solve(prob1.1, &prob1.0);
            let prob2_sol = solver.solve(prob2.1, &prob2.0);

            println!("the numerical solution of $2e^x - 3 = 0$ is: {}", prob1_sol);
            println!("with direct calc: {}", (1.5_f64).ln());
            println!("the numerical solution of $2x^2 - 5x + 3 = 0$ is: {}", prob2_sol);
            println!("with direct calc: {} or {}", ((5. + 1.)/4.) , (3./4.));

            println!("cloned sol: {}", cloned_and_change_cond_sol);

            assert!(prob1.0.calc(prob1_sol) < 1e-6);
            assert!(prob2.0.calc(prob2_sol) < 1e-6)
        }

        #[test]
        fn solver_iter() {
                let iter_fn = |state: &NewtonState, problem: &Equation| {
                let x = state.0;
                let dx = problem.diff(x);
                let fx = problem.calc(x);

                let next_x = x - (fx / dx);

                NewtonState(next_x)
            };

            let term_cond = |state: &NewtonState, problem: &Equation| {
                let epsilon = 1e-6;
                problem.calc(state.0).abs() < epsilon
            };

            let solver = Solver::new(iter_fn, term_cond);

            let prob1 = (Equation::Exp { a: 2., k: 3. }, 2.);

            let prob2 = (Equation::Square { a: 2., b: -5., c: 3. }, 6.);

            {
                let iter = solver.clone().into_iter(prob1.1, &prob1.0);
                let mut counter = 0usize;
                
                for state in iter {
                    counter += 1;
                    if counter == 5 {
                        println!("after 5 times iter, the f(x) = {}", (prob1.0).calc(state.0));
                        break;
                    }
                }
            }

            //solver.init();

            {
                let iter = solver.into_iter(prob2.1, &prob2.0);
                let mut counter = 0usize;
                
                for state in iter {
                    counter += 1;
                    if counter == 5 {
                        println!("after 5 times iter, the f(x) = {}", (prob2.0).calc(state.0));
                        break;
                    }
                }
            }
        }
    }

    mod test_leak {
        use std::time::Duration;

        use crate::{utils, Solver};
        

        #[test]
        fn solve() {
            let iter_fn = |state: &utils::debug::VisibleDrop, _: &()| {
                utils::debug::VisibleDrop::new(state.get().wrapping_add(1))
            };

            let term_cond = |state: &utils::debug::VisibleDrop, _: &()| {
                state.get() == 2
            };

            let mut solver = Solver::new(iter_fn, term_cond);

            println!("solve");

            solver.solve(0, &());


            println!("iter");
            for _ in solver.clone().into_iter(0, &()) {
                println!("do iter")
            }

            println!("solve with timeout");
            let mut loop_solver = solver.change_term_cond(|_,_| false);

            let timeout = Duration::from_nanos(1000);

            let _ = loop_solver.solve_with_timeout(0, &(), timeout);
        }
    }
}
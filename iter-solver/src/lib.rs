#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate self as iter_solver;

pub use iter_state_derive::IterState;

use core::marker::PhantomData;

#[cfg(feature = "std")]
use std::time::Duration;

#[cfg(feature = "std")]
use std::time::Instant;

#[cfg(feature = "std")]
use crate::error::TimeOut;

use crate::error::MaxIterationReached;

pub mod error;

#[cfg(test)]
mod build_test;

/// Intermediate state of an iterative algorithm.
///
/// For many iterative algorithms, the intermediate state often requires a more complex structure to describe compared to the initial iteration value and the solution.
///
/// In practical iterative algorithms, it is often necessary to perform some simple computations or obtain algorithm metadata after obtaining the solution.
///
/// Therefore, IterState allows you to customize the intermediate state and separate the abstractions of value and solution.
///
/// If you expect the simplest behavior, this crate has already implemented `IterState` for basic data types `i*`, `u*`, and `f*`, where their associated types `Value` and `Solution` are themselves.
/// 
/// Additionally, you can use the macro `#[derive(IterState)]` to derive `IterState`. In this case, the associated types [`IterState::Value`] and [`IterState::Solution`] will be set to `Self`. Moreover, similar to the implementation for basic data types, the [`IterState::init_from_value`] method will simply return the parameter directly, and the [`IterState::into_sol`] method will directly return `self`.
pub trait IterState: Sized {
    /// Type representing the value during iteration (e.g., intermediate computation results).
    type Value;

    /// Type representing the final solution.  
    type Solution;  

    /// Initializes the state from an initial value.  
    fn init_from_value(initial_point: Self::Value) -> Self;  

    /// Converts the current state into the solution.  
    fn into_sol(self) -> Self::Solution;  
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
                fn into_sol(self) -> Self::Solution {
                    self
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
    IterFn: Fn(State, &Problem) -> State,
    TermFn: Fn(&State, &Problem) -> bool,
{
    /// Intermediate state of the solver (uninitialized at the start)
    state: PhantomData<State>,
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
    IterFn: Fn(State, &Problem) -> State,
    TermFn: Fn(&State, &Problem) -> bool
{
    /// Creates a Solver instance with the specified methods.
    ///
    /// Parameter `iter_fn` defines the state iteration rule for the Solver.
    ///
    /// Parameter `term_cond` specifies that the Solver stops iterating if and only if
    /// the condition is met: `term_cond` returns `true` in the current state.
    pub fn new(
        iter_fn: IterFn,
        term_cond: TermFn
    ) -> Self {
        Self { 
            state: PhantomData::<State>, 
            problem: PhantomData::<Problem>, 
            iter_fn: iter_fn, 
            term_cond: term_cond 
        }
    }

    /// Solves the problem by executing iterative logic with the given initial value and specific problem.
    ///
    /// # Note
    /// If the algorithm defined by the solver contains logical errors, 
    /// the solve function may enter an infinite loop. 
    /// To avoid this, try [`Solver::solve_with_max_iterations`] or [`Solver::solve_with_timeout`]. If you need more flexible error handling, try [`Solver::solve_with_error`].
    pub fn solve(&self, initial_point: State::Value, problem: &Problem) -> State::Solution {
        // init state
        let initial_state = State::init_from_value(initial_point);
        let mut state = initial_state;

        if (self.term_cond)(&state, problem) {
            return state.into_sol();
        }
        
        // do iter
        loop {
            //let state = unsafe { self.state.assume_init_mut() };

            state = (self.iter_fn)(state, problem);

            // check termination cond
            if (self.term_cond)(&state, problem) {
                break;
            }
        }
        
        let final_state = state;

        let sol = final_state.into_sol();

        //unsafe { self.state.assume_init_drop(); }

        sol
    }


    /// A solution method with a maximum iteration limit.
    /// If the termination condition is met before reaching the maximum number of iterations, it returns an [`Ok`] value with the type [`IterState::Solution`].
    /// Otherwise, it returns an [`Err`] with [`error::MaxIterationReached`].
    /// 
    /// # Example
    /// ```
    /// use iter_solver::Solver;
    /// 
    /// // define a never stop solver
    /// let loop_solver = Solver::new(|_state: f64, _: &()| {_state}, |_: &f64, _: &()| {false});
    /// let try_solve = loop_solver.solve_with_max_iterations(0.0, &(), 10);
    /// 
    /// assert!(try_solve.is_err());
    /// ```
    /// 
    /// If you need more flexible error handling, try [`Solver::solve_with_error`].
    pub fn solve_with_max_iterations(
        &self,
        initial_point: State::Value,
        problem: &Problem,
        max_iteration: u64
    ) -> Result<State::Solution, MaxIterationReached<State>> {
        let mut reach_max = true;

        // init state
        let initial_state = State::init_from_value(initial_point);
        let mut state = initial_state;     

        if (self.term_cond)(&state, problem) {
            return Ok(state.into_sol());
        }   
        
        for _iteration in 0..max_iteration {
            //state = unsafe { self.state.assume_init_mut() };

            state = (self.iter_fn)(state, problem);

            // check termination cond
            if (self.term_cond)(&state, problem) {
                reach_max = false;
                break;
            }
        }

        let ret_res = if !reach_max {
            let final_state = state;
            let sol = final_state.into_sol();
            //unsafe { self.state.assume_init_drop(); }
            Ok(sol)
        } else {
            let final_state = state;
            Err(MaxIterationReached{
                max_iteration: max_iteration, 
                final_state: final_state
            })
        };

        ret_res
    }

    #[cfg(feature = "std")]
    /// A solution method with a time limit.
    /// If the termination condition is met before reaching the timeout duration elapses, it returns an [`Ok`] value with the type [`IterState::Solution`].
    /// Otherwise, it returns an [`Err`] with [`error::TimeOut`].
    /// 
    /// # Example
    /// ```
    /// use iter_solver::Solver;
    /// use std::time::Duration;
    /// 
    /// // define a never stop solver
    /// let loop_solver = Solver::new(|_state: f64, _: &()| {_state}, |_: &f64, _: &()| {false});
    /// let try_solve = loop_solver.solve_with_timeout(0.0, &(), Duration::from_secs(1));
    /// 
    /// assert!(try_solve.is_err());
    /// ```
    /// 
    /// If you need more flexible error handling, try [`Solver::solve_with_error`].
    pub fn solve_with_timeout(
        &self,
        initial_point: State::Value,
        problem: &Problem,
        timeout: Duration        
    ) -> Result<State::Solution, TimeOut<State>> {
        let start_time = Instant::now();
        let mut is_timeout = true;
        
        // init state
        let initial_state = State::init_from_value(initial_point);
        let mut state = initial_state;

        if (self.term_cond)(&state, problem) {
            return Ok(state.into_sol());
        }
        
        // do iter
        loop {
            //state = unsafe { self.state.assume_init_mut() };

            state = (self.iter_fn)(state, problem);

            if start_time.elapsed() > timeout {
                break;
            }

            // check termination cond
            if (self.term_cond)(&state, problem) {
                is_timeout = false;
                break;
            }
        }

        if !is_timeout {
            let final_state = state;

            let sol = final_state.into_sol();

            //unsafe { self.state.assume_init_drop(); }

            Ok(sol)                
        } else {
            let final_state = state; //unsafe { self.state.assume_init_read() };
            Err(TimeOut { timeout: timeout, final_state: final_state })            
        }

    }

    /// Performs iterative solving with custom error handling, allowing early termination.
    ///
    /// This method executes an iterative solving process, but before each iteration,
    /// it invokes the provided `check_fn` with the current state and the problem reference.
    /// If `check_fn` returns `Ok(())`, the iteration continues until the stopping criteria
    /// are met, returning [`Ok`] with the final solution. If `check_fn` returns `Err(e)`, the
    /// iteration stops immediately and returns `Err(e)`.
    ///
    /// The key feature is **flexible error type customization** – `E` can be any type
    /// that suits your error-handling needs (e.g., a simple `&'static str`, a custom enum,
    /// or a structured error type). This allows you to:
    /// - Embed domain-specific failure semantics directly into the solving flow.
    /// - Propagate rich error information without boxing or trait objects.
    /// - Maintain full control over error kinds and context.
    /// 
    /// # Example
    /// ```
    /// use iter_solver::Solver;
    ///         
    /// let check_fn = |float: &f64, _: &()| {
    ///     if float.is_infinite() {
    ///         return Err("Inf Error");
    ///     } else if float.is_nan() {
    ///        return Err("NaN Error");
    ///     }
    ///     Ok(())
    /// };
    ///
    /// let solver = Solver::new(
    ///    |f, _| f * 2.0, // 2^n -> Inf
    ///    |_,_| {false} // never stop
    /// );
    ///
    /// let result = solver.solve_with_error(1.0, &(), check_fn);
    ///
    /// assert!(result.is_err());
    /// println!("{}", result.unwrap_err()) // print "Inf Error"
    /// ```
    pub fn solve_with_error<E, F>(
        &self,
        initial_point: State::Value,
        problem: &Problem,
        check_fn: F
    ) -> Result<State::Solution, E>
    where 
        F: Fn(&State, &Problem) -> Result<(), E>
    {
        // init state
        let initial_state = State::init_from_value(initial_point);
        let mut state = initial_state;

        check_fn(&state, problem)?;
        if (self.term_cond)(&state, problem) {
            return Ok(state.into_sol());
        }
        
        // do iter
        loop {
            //let state = unsafe { self.state.assume_init_mut() };

            state = (self.iter_fn)(state, problem);

            check_fn(&state, problem)?;
            // check termination cond
            if (self.term_cond)(&state, problem) {
                break;
            }
        }
        
        let final_state = state;

        let sol = final_state.into_sol();

        //unsafe { self.state.assume_init_drop(); }

        Ok(sol)
    }



    /// Consumes `self` and returns a new [`Solver`] with the given new termination condition.
    pub fn with_term_cond<NewTermCond>(self, new_cond: NewTermCond) -> Solver<State, Problem, IterFn, NewTermCond> 
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
   IterFn: Fn(State, &Problem) -> State + Clone,
   TermFn: Fn(&State, &Problem) -> bool + Clone
{
    fn clone(&self) -> Self {
        Self { state: PhantomData::<State>, problem: PhantomData::<Problem>, iter_fn: self.iter_fn.clone(), term_cond: self.term_cond.clone() }
    }
}






#[cfg(test)]
mod test {
    use crate::Solver;

    mod newton {
        use crate::{IterState, Solver};

        fn f_and_df(x: f64) -> (f64, f64) {
            let fx = x.exp() - 1.5;
            let dfx = x.exp();
            (fx, dfx)            
        }
        #[test]
    fn show() {
        let iter_fn = |state: f64, problem: &fn(f64) -> (f64, f64)| {
            let x_n = state;
            let (fx, dfx) = problem(x_n);
            x_n - (fx / dfx)
        };

        let term_cond = |state: &f64, problem: &fn(f64) -> (f64, f64)| {
            let (fx, _) = problem(*state);
            fx.abs() < 1e-6
        };

        let solver = Solver::new(iter_fn, term_cond);

        let solution = solver.solve(1.5, &(f_and_df as fn(f64) -> (f64, f64)));

        println!("solver's solution: {}", solution);  
        println!("use std function ln: {}", 1.5_f64.ln());

        // solver's solution: 0.4054651081202111
        // use std function ln: 0.4054651081081644
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
        
            fn into_sol(self) -> Self::Solution {
                self.0
            }
        }

        #[test]
        fn test() {
            let iter_fn = |state: NewtonState, problem: &Equation| {
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

            let  solver = Solver::new(iter_fn, term_cond);

            let  solver1 = solver.clone().with_term_cond(|state, equation| {
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

        
            
    }

    
    #[test]
    fn test_with_error() {
        let check_fn = |float: &f64, _: &()| {
            if float.is_infinite() {
                return Err("Inf Error");
            } else if float.is_nan() {
                return Err("NaN Error");
            }
            Ok(())
        };

        let solver = Solver::new(
            |f, _| f * 2.0, // 2^n -> Inf
            |_,_| {false} // never stop
        );

        let result = solver.solve_with_error(1.0, &(), check_fn);

        assert!(result.is_err());
        println!("{}", result.unwrap_err()) // Inf Error
    }
    

    mod guard {
        use std::time::Duration;

        use crate::Solver;

        #[test]
        fn test() {
            
            // define a never stop solver
            let loop_solver = Solver::new(|_state: f64, _: &()| {_state}, |_: &f64, _: &()| {false});
            let try_solve = loop_solver.solve_with_timeout(0.0, &(), Duration::from_secs(1));
            
            assert!(try_solve.is_err());

            let try_solve = loop_solver.solve_with_max_iterations(0.0, &(), 10);
            assert!(try_solve.is_err());

        }
    }


    mod derive_test {
        use crate::IterState;

        #[derive(PartialEq, Eq , IterState, Debug)]
        struct State(Vec<u8>, Box<String>);

        #[test]
        fn test_derive() {
            let vec1 = vec![0,12, 39];
            let boxed_str = Box::new("some str".to_string());
            let value = State(vec1.clone(), boxed_str.clone());
            let state = State::init_from_value(value);
            assert_eq!(vec1, state.0);
            let final_s = state.into_sol();
            assert_eq!(final_s.1, boxed_str);
        }
    }
}
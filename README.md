# A General Iterative Algorithm Framework
iter-solver is a flexible and general iterative algorithm framework that allows users to customize iteration logic and termination conditions, providing a unified abstraction for different problems and algorithms.
# Example
The following program defines a simple Newton iterative solver to compute ln(1.5):
```no_run
use iter_solver::Solver;

fn f_and_df(x: f64) -> (f64, f64) {
    let fx = x.exp() - 1.5;
    let dfx = x.exp();
    (fx, dfx) 
}

fn main() {
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
```
# Installation
Add the driver to your `Cargo.toml` dependencies:
```ignor
[dependencies]
iter-solver = "0.2.0"
```
or add it directly from the terminal:
```ignor
cargo add iter-solver
```

# License
[MIT License](https://opensource.org/license/MIT).
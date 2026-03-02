//! A derive macro for `IterState` trait of the crate [iter-solver](https://crates.io/crates/iter-solver).


use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

/// A derive macro that automatically derives the `IterState` trait for structs that satisfy the `Clone` trait. 
/// 
/// When using this macro to implement the trait, the associated types `Value` and `Solution` will both be set to `Self`, the `init_from_value` method will simply return the parameter directly, and the `to_sol` method will directly clone `self` and return it.
#[proc_macro_derive(IterState)]
pub fn iter_state_trait_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ::iter_solver::IterState for #name #ty_generics #where_clause
        where
            Self: Clone,
        {
            type Value = Self;
            type Solution = Self;

            fn init_from_value(initial_point: Self::Value) -> Self {
                initial_point
            }

            fn to_sol(&self) -> Self::Solution {
                <Self as Clone>::clone(self)
            }
        }
    };

    TokenStream::from(expanded)
}
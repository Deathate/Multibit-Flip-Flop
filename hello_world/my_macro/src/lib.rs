// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }

// extern crate proc_macro;

// use proc_macro::TokenStream;
// use quote::quote;
// use syn::{parse_macro_input, ItemStruct};

// #[proc_macro]
// pub fn duplicate_struct(input: TokenStream) -> TokenStream {
//     let input_struct = parse_macro_input!(input as ItemStruct);
//     let struct_name = &input_struct.ident;
//     let fields = &input_struct.fields;

//     let new_struct_name = syn::Ident::new("B", struct_name.span());
//     let output = quote! {
//         #input_struct

//         struct #new_struct_name {
//             #fields
//         }
//     };

//     output.into()
// }
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, ItemStruct};
#[proc_macro_derive(Similar)]
pub fn derive_similar(input: TokenStream) -> TokenStream {
    // Parse the input struct
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = input.ident; // Name of the original struct

    // Generate the name for the new struct (e.g., append "Similar")
    let new_struct_name = syn::Ident::new(&format!("{}Similar", struct_name), struct_name.span());

    // Extract the struct fields
    let fields = if let syn::Data::Struct(data) = input.data {
        data.fields
    } else {
        panic!("#[derive(Similar)] can only be used on structs");
    };

    // Generate the new struct with `#[derive(Debug)]`
    let output = quote! {
        #[derive(Debug)]
        pub struct #new_struct_name {
            #fields
        }
    };

    output.into()
}
#[proc_macro]
pub fn duplicate_struct(input: TokenStream) -> TokenStream {
    let input_struct = parse_macro_input!(input as ItemStruct);
    let struct_name = &input_struct.ident;
    let fields = &input_struct.fields;

    let new_struct_name = syn::Ident::new("B", struct_name.span());
    let output = quote! {
        #input_struct

        struct #new_struct_name {
            #fields
        }
    };

    output.into()
}

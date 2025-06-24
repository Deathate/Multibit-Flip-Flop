use proc_macro::TokenStream;
mod forward_methods_macro;
mod shared_weak_macro;
#[proc_macro_derive(SharedWeakWrappers, attributes(skip, hash))]
pub fn shared_weak_wrappers(input: TokenStream) -> TokenStream {
    shared_weak_macro::expand(input)
}

#[proc_macro_attribute]
pub fn forward_methods(_attr: TokenStream, input: TokenStream) -> TokenStream {
    forward_methods_macro::expand(input)
}

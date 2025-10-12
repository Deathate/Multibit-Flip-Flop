use proc_macro::TokenStream;
mod forward_methods_macro;
mod forward_methods_macro_thread;
mod forward_methods_macro_rwlock;
mod shared_weak_macro;
mod shared_weak_macro_thread;
mod shared_weak_macro_thread_rwlock;

#[cfg(not(feature = "threaded"))]
#[proc_macro_derive(SharedWeakWrappers, attributes(skip, hash))]
pub fn shared_weak_wrappers(input: TokenStream) -> TokenStream {
    shared_weak_macro::expand(input)
}
#[cfg(not(feature = "threaded"))]
#[proc_macro_attribute]
pub fn forward_methods(_attr: TokenStream, input: TokenStream) -> TokenStream {
    forward_methods_macro::expand(input)
}
// #[cfg(feature = "threaded")]
// #[proc_macro_derive(SharedWeakWrappers, attributes(skip, hash))]
// pub fn shared_weak_wrappers_thread(input: TokenStream) -> TokenStream {
//     shared_weak_macro_thread::expand(input)
// }
// #[cfg(feature = "threaded")]
// #[proc_macro_attribute]
// pub fn forward_methods(_attr: TokenStream, input: TokenStream) -> TokenStream {
//     forward_methods_macro_thread::expand(input)
// }
#[cfg(feature = "threaded")]
#[proc_macro_derive(SharedWeakWrappers, attributes(skip, hash))]
pub fn shared_weak_wrappers_rw(input: TokenStream) -> TokenStream {
    shared_weak_macro_thread_rwlock::expand(input)
}
#[cfg(feature = "threaded")]
#[proc_macro_attribute]
pub fn forward_methods(_attr: TokenStream, input: TokenStream) -> TokenStream {
    forward_methods_macro_rwlock::expand(input)
}

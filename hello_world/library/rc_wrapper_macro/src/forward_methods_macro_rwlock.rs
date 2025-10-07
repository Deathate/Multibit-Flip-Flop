use quote::{ToTokens, format_ident, quote};
use syn::{ImplItem, ItemImpl, ReturnType, parse_macro_input};

pub fn expand(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input_clone = input.clone();
    let item_impl = parse_macro_input!(input as ItemImpl);
    let self_ty = &item_impl.self_ty;
    let struct_ident = if let syn::Type::Path(p) = &**self_ty {
        &p.path.segments.last().unwrap().ident
    } else {
        panic!("Unsupported type for impl block");
    };

    let shared_name = format_ident!("Shared{}", struct_ident);
    let weak_name = format_ident!("Weak{}", struct_ident);

    let mut shared_methods = vec![];
    let mut weak_methods = vec![];

    for item in &item_impl.items {
        if let ImplItem::Fn(method) = item {
            let method_name = &method.sig.ident;
            let method_attrs = &method.attrs;
            let method_vis = &method.vis;
            let output = &method.sig.output;
            let method_generics = &method.sig.generics; // <T, U> part on fn
            let method_where = &method.sig.generics.where_clause; // where ... on fn

            let is_reference = matches!(output, ReturnType::Type(_, ty) if matches!(**ty, syn::Type::Reference(_)));

            // only wrap public fns
            if !method_vis.to_token_stream().to_string().contains("pub") {
                continue;
            }

            // collect non-receiver inputs
            let inputs: Vec<_> = method
                .sig
                .inputs
                .iter()
                .filter(|arg| !matches!(arg, syn::FnArg::Receiver(_)))
                .collect();

            let args: Vec<_> = inputs
                .iter()
                .filter_map(|arg| match arg {
                    syn::FnArg::Typed(pat_type) => Some(&pat_type.pat),
                    _ => None,
                })
                .collect();

            let call_args = quote! { #(#args),* };

            // figure out receiver mutability to pick read()/write()
            if let Some(receiver) = method.sig.receiver() {
                let wants_write = receiver.mutability.is_some();

                // helpers for the guard type & map fn when returning references
                let (guard_acquire, read_guard_ty, write_guard_ty, read_map, write_map) = (
                    if wants_write {
                        quote! { write() }
                    } else {
                        quote! { read() }
                    },
                    quote! { parking_lot::RwLockReadGuard },
                    quote! { parking_lot::RwLockWriteGuard },
                    quote! { parking_lot::RwLockReadGuard::map },
                    quote! { parking_lot::RwLockWriteGuard::map },
                );

                let method_body = if !is_reference {
                    // Non-reference return: call through the appropriate guard and return value
                    if wants_write {
                        quote! {
                            #(#method_attrs)*
                            #method_vis fn #method_name #method_generics (&self, #(#inputs),*) #output #method_where {
                                let mut guard = self.0.write();
                                // Call the inner method on &mut *guard
                                guard.#method_name(#call_args)
                            }
                        }
                    } else {
                        quote! {
                            #(#method_attrs)*
                            #method_vis fn #method_name #method_generics (&self, #(#inputs),*) #output #method_where {
                                let guard = self.0.read();
                                // Call the inner method on &*guard
                                guard.#method_name(#call_args)
                            }
                        }
                    }
                } else {
                    // Reference return: map the guard to the returned ref
                    let (inner_ty, returns_mut) = match output {
                        ReturnType::Type(_, ty) => match &**ty {
                            syn::Type::Reference(r) => (&r.elem, r.mutability.is_some()),
                            _ => unreachable!("Checked is_reference above"),
                        },
                        ReturnType::Default => unreachable!("reference return cannot be default"),
                    };

                    if returns_mut {
                        // &mut T return -> must acquire write lock and map to MappedRwLockWriteGuard<'_, T>
                        quote! {
                            #(#method_attrs)*
                            #method_vis fn #method_name #method_generics (&self, #(#inputs),*)
                                -> parking_lot::MappedRwLockWriteGuard<'_, #inner_ty> #method_where
                            {
                                let guard = self.0.write();
                                #write_map(guard, |a| a.#method_name(#call_args))
                            }
                        }
                    } else {
                        // &T return -> read lock and MappedRwLockReadGuard<'_, T>
                        quote! {
                            #(#method_attrs)*
                            #method_vis fn #method_name #method_generics (&self, #(#inputs),*)
                                -> parking_lot::MappedRwLockReadGuard<'_, #inner_ty> #method_where
                            {
                                let guard = self.0.read();
                                #read_map(guard, |a| a.#method_name(#call_args))
                            }
                        }
                    }
                };

                shared_methods.push(method_body);

                // For Weak*: only generate wrappers for non-reference returns (like the original)
                if !is_reference {
                    if wants_write {
                        weak_methods.push(quote! {
                            #(#method_attrs)*
                            #method_vis fn #method_name #method_generics (&self, #(#inputs),*) #output #method_where {
                                let upgraded = self.0.upgrade().expect("Weak pointer expired");
                                let mut guard = upgraded.write();
                                guard.#method_name(#call_args)
                            }
                        });
                    } else {
                        weak_methods.push(quote! {
                            #(#method_attrs)*
                            #method_vis fn #method_name #method_generics (&self, #(#inputs),*) #output #method_where {
                                let upgraded = self.0.upgrade().expect("Weak pointer expired");
                                let guard = upgraded.read();
                                guard.#method_name(#call_args)
                            }
                        });
                    }
                }
            }
        }
    }

    let input_clone: proc_macro2::TokenStream = input_clone.into();
    let expanded = quote! {
        #input_clone

        impl #shared_name {
            #(#shared_methods)*
        }

        impl #weak_name {
            #(#weak_methods)*
        }
    };

    proc_macro::TokenStream::from(expanded)
}

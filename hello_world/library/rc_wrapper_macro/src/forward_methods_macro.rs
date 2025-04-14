use quote::{ToTokens, format_ident, quote};
use syn::{ImplItem, ItemImpl, Receiver, ReturnType, parse_macro_input};

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
            let is_reference = match output {
                ReturnType::Type(_, ty) => matches!(**ty, syn::Type::Reference(_)),
                ReturnType::Default => false,
            };

            if !method_vis.to_token_stream().to_string().contains("pub") {
                continue;
            }

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
            let output_ty = match output {
                ReturnType::Type(_, ty) => Some(ty),
                ReturnType::Default => None,
            };
            /// Expands the receiver (`self`, `&self`, `&mut self`, etc.) into a TokenStream.
            fn expand_self_arg(receiver: &Receiver) -> proc_macro2::TokenStream {
                match (&receiver.reference, &receiver.mutability) {
                    (Some((_, Some(lifetime))), Some(_)) => quote! { & #lifetime mut self },
                    (Some((_, Some(lifetime))), None) => quote! { & #lifetime self },
                    (Some(_), Some(_)) => quote! { &mut self },
                    (Some(_), None) => quote! { &self },
                    (None, Some(_)) => quote! { mut self },
                    (None, None) => quote! { self },
                }
            }
            if let Some(receiver) = method.sig.receiver() {
                let is_mutable = receiver.mutability.is_some();
                let self_arg = expand_self_arg(receiver);
                let borrow_method = if is_mutable {
                    quote! { borrow_mut() }
                } else {
                    quote! { borrow() }
                };

                let method_body = if !is_reference {
                    // Non-reference return: direct method call
                    quote! {
                        #[inline(always)]
                        #(#method_attrs)*
                        #method_vis fn #method_name(#self_arg, #(#inputs),*) #output {
                            self.0.#borrow_method.#method_name(#call_args)
                        }
                    }
                } else {
                    // Reference return: wrap with Ref::map
                    let inner_ty =
                        match &**output_ty.expect("Output type must exist for reference return") {
                            syn::Type::Reference(syn::TypeReference { elem, .. }) => elem,
                            _ => unreachable!("Checked is_reference above"),
                        };
                    quote! {
                        #[inline(always)]
                        #(#method_attrs)*
                        #method_vis fn #method_name(#self_arg, #(#inputs),*) -> std::cell::Ref<#inner_ty> {
                            std::cell::Ref::map(self.0.#borrow_method, |a| a.#method_name(#call_args))
                        }
                    }
                };

                shared_methods.push(method_body);

                if !is_reference {
                    // Only implement weak method for non-reference return
                    weak_methods.push(quote! {
                        #[inline(always)]
                        #(#method_attrs)*
                        #method_vis fn #method_name(#self_arg, #(#inputs),*) #output {
                            self.0.upgrade().unwrap().#borrow_method.#method_name(#call_args)
                        }
                    });
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

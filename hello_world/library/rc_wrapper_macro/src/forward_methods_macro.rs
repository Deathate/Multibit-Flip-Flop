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

            if let Some(receiver) = method.sig.receiver() {
                if receiver.mutability.is_some() {
                    shared_methods.push(quote! {
                        #(#method_attrs)*
                        #method_vis fn #method_name(&mut self, #(#inputs),*) #output {
                            self.0.borrow_mut().#method_name(#call_args)
                        }
                    });
                    weak_methods.push(quote! {
                        #(#method_attrs)*
                        #method_vis fn #method_name(&mut self, #(#inputs),*) #output {
                            self.0.upgrade().unwrap().borrow_mut().#method_name(#call_args)
                        }
                    });
                } else {
                    shared_methods.push(quote! {
                        #(#method_attrs)*
                        #method_vis fn #method_name(&mut self, #(#inputs),*) #output {
                            self.0.borrow().#method_name(#call_args)
                        }
                    });
                    weak_methods.push(quote! {
                        #(#method_attrs)*
                        #method_vis fn #method_name(&self, #(#inputs),*) #output {
                            self.0.upgrade().unwrap().borrow().#method_name(#call_args)
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

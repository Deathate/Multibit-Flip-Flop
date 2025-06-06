use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Attribute, FnArg, ItemImpl, ItemStruct, Result, Token, braced,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

struct RcWrapperInput {
    attrs: Vec<Attribute>,
    vis: syn::Visibility,
    item_struct: ItemStruct,
    item_impl: ItemImpl,
}

impl Parse for RcWrapperInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: syn::Visibility = input.parse()?;
        let item_struct: ItemStruct = input.parse()?;
        let item_impl: ItemImpl = input.parse()?;
        Ok(Self {
            attrs,
            vis,
            item_struct,
            item_impl,
        })
    }
}

#[proc_macro]
pub fn define_shared_wrapper(input: TokenStream) -> TokenStream {
    let RcWrapperInput {
        attrs,
        vis,
        item_struct,
        item_impl,
    } = parse_macro_input!(input as RcWrapperInput);

    let struct_name = &item_struct.ident;
    let wrapper_name = format_ident!("Shared{}", struct_name);

    let methods = item_impl
        .items
        .iter()
        .filter_map(|item| {
            if let syn::ImplItem::Fn(method) = item {
                // Skip methods that don't have a self parameter
                if method.sig.inputs.is_empty()
                    || !method
                        .sig
                        .inputs
                        .iter()
                        .any(|arg| matches!(arg, syn::FnArg::Receiver(_)))
                {
                    return None;
                }

                let method_name = &method.sig.ident;
                let method_attrs = &method.attrs;
                let method_vis = &method.vis;
                // check if the method is pub
                if !method_vis.to_token_stream().to_string().contains("pub") {
                    return None;
                }
                let output = &method.sig.output;

                // Collect all non-self parameters
                let inputs: Vec<_> = method
                    .sig
                    .inputs
                    .iter()
                    .filter(|arg| !matches!(arg, syn::FnArg::Receiver(_)))
                    .collect();

                // Collect just the argument names (without types) for the method call
                let args: Vec<_> = inputs
                    .iter()
                    .filter_map(|arg| match arg {
                        syn::FnArg::Typed(pat_type) => Some(&pat_type.pat),
                        _ => None,
                    })
                    .collect();

                let call_args = quote! { #(#args),* };

                // Check the receiver type
                if let Some(receiver) = method.sig.receiver() {
                    if receiver.mutability.is_some() {
                        // &mut self method
                        Some(quote! {
                            #(#method_attrs)*
                            #method_vis fn #method_name(&mut self, #(#inputs),*) #output {
                                self.inner.borrow_mut().#method_name(#call_args)
                            }
                        })
                    } else {
                        // &self method
                        Some(quote! {
                            #(#method_attrs)*
                            #method_vis fn #method_name(&self, #(#inputs),*) #output {
                                self.inner.borrow().#method_name(#call_args)
                            }
                        })
                    }
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let expanded = quote! {
        #(#attrs)*
        #vis #item_struct

        #item_impl

        pub struct #wrapper_name {
            inner: std::rc::Rc<std::cell::RefCell<#struct_name>>,
        }

        impl #wrapper_name {
            pub fn new(value: #struct_name) -> Self {
                Self {
                    inner: std::rc::Rc::new(std::cell::RefCell::new(value)),
                }
            }

            pub fn borrow(&self) -> std::cell::Ref<#struct_name> {
                self.inner.borrow()
            }

            pub fn borrow_mut(&self) -> std::cell::RefMut<#struct_name> {
                self.inner.borrow_mut()
            }

            pub fn get_ref(&self) -> &std::rc::Rc<std::cell::RefCell<#struct_name>> {
                &self.inner
            }

            #(#methods)*
        }
        impl Clone for #wrapper_name {
            fn clone(&self) -> Self {
                Self {
                    inner: std::rc::Rc::clone(&self.inner),
                }
            }
        }
        impl fmt::Debug for #wrapper_name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!(#wrapper_name))
                    .field("inner", &self.borrow())
                    .finish()
            }
        }
        impl From<std::rc::Rc<std::cell::RefCell<#struct_name>>> for #wrapper_name {
            fn from(inner: std::rc::Rc<std::cell::RefCell<#struct_name>>) -> Self {
                Self { inner }
            }
        }
        impl From<#wrapper_name> for std::rc::Rc<std::cell::RefCell<#struct_name>> {
            fn from(wrapper: #wrapper_name) -> Self {
                wrapper.inner
            }
        }
    };
    // eprintln!("Expanded code: {}", expanded);

    TokenStream::from(expanded)
}

#[proc_macro]
pub fn define_shared_weak_wrappers(input: TokenStream) -> TokenStream {
    let RcWrapperInput {
        attrs,
        vis,
        item_struct,
        item_impl,
    } = parse_macro_input!(input as RcWrapperInput);

    let struct_name = &item_struct.ident;
    let shared_name = format_ident!("Shared{}", struct_name);
    let weak_name = format_ident!("Weak{}", struct_name);

    // Extract public &self / &mut self methods from the impl
    let methods = item_impl
        .items
        .iter()
        .filter_map(|item| {
            if let syn::ImplItem::Fn(method) = item {
                if method.sig.inputs.is_empty()
                    || !method
                        .sig
                        .inputs
                        .iter()
                        .any(|arg| matches!(arg, syn::FnArg::Receiver(_)))
                {
                    return None;
                }

                let method_name = &method.sig.ident;
                let method_attrs = &method.attrs;
                let method_vis = &method.vis;

                if !method_vis.to_token_stream().to_string().contains("pub") {
                    return None;
                }

                let output = &method.sig.output;
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
                        Some(quote! {
                            #(#method_attrs)*
                            #method_vis fn #method_name(&mut self, #(#inputs),*) #output {
                                self.inner.borrow_mut().#method_name(#call_args)
                            }
                        })
                    } else {
                        Some(quote! {
                            #(#method_attrs)*
                            #method_vis fn #method_name(&self, #(#inputs),*) #output {
                                self.inner.borrow().#method_name(#call_args)
                            }
                        })
                    }
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let methods_weak = item_impl
        .items
        .iter()
        .filter_map(|item| {
            if let syn::ImplItem::Fn(method) = item {
                if method.sig.inputs.is_empty()
                    || !method
                        .sig
                        .inputs
                        .iter()
                        .any(|arg| matches!(arg, syn::FnArg::Receiver(_)))
                {
                    return None;
                }

                let method_name = &method.sig.ident;
                let method_attrs = &method.attrs;
                let method_vis = &method.vis;

                if !method_vis.to_token_stream().to_string().contains("pub") {
                    return None;
                }

                let output = &method.sig.output;
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
                        Some(quote! {
                            #(#method_attrs)*
                            #method_vis fn #method_name(&mut self, #(#inputs),*) #output {
                                self.inner.upgrade().unwrap().borrow_mut().#method_name(#call_args)
                            }
                        })
                    } else {
                        Some(quote! {
                            #(#method_attrs)*
                            #method_vis fn #method_name(&self, #(#inputs),*) #output {
                                self.inner.upgrade().unwrap().borrow().#method_name(#call_args)
                            }
                        })
                    }
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let expanded = quote! {
        #(#attrs)*
        #vis #item_struct

        #item_impl

        /// Strong wrapper: Shared<T>
        pub struct #shared_name {
            inner: std::rc::Rc<std::cell::RefCell<#struct_name>>,
        }

        impl #shared_name {
            pub fn new(value: #struct_name) -> Self {
                Self {
                    inner: std::rc::Rc::new(std::cell::RefCell::new(value)),
                }
            }

            pub fn borrow(&self) -> std::cell::Ref<#struct_name> {
                self.inner.borrow()
            }

            pub fn borrow_mut(&self) -> std::cell::RefMut<#struct_name> {
                self.inner.borrow_mut()
            }

            pub fn get_ref(&self) -> &std::rc::Rc<std::cell::RefCell<#struct_name>> {
                &self.inner
            }

            pub fn downgrade(&self) -> #weak_name {
                #weak_name {
                    inner: std::rc::Rc::downgrade(&self.inner),
                }
            }

            #(#methods)*
        }

        impl Clone for #shared_name {
            fn clone(&self) -> Self {
                Self {
                    inner: std::rc::Rc::clone(&self.inner),
                }
            }
        }

        impl std::fmt::Debug for #shared_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!(#shared_name))
                    .field("inner", &self.borrow())
                    .finish()
            }
        }

        impl From<std::rc::Rc<std::cell::RefCell<#struct_name>>> for #shared_name {
            fn from(inner: std::rc::Rc<std::cell::RefCell<#struct_name>>) -> Self {
                Self { inner }
            }
        }

        impl From<#shared_name> for std::rc::Rc<std::cell::RefCell<#struct_name>> {
            fn from(wrapper: #shared_name) -> Self {
                wrapper.inner
            }
        }

        /// Weak wrapper: Weak<T>
        pub struct #weak_name {
            inner: std::rc::Weak<std::cell::RefCell<#struct_name>>,
        }

        impl #weak_name {
            pub fn new(value: &std::rc::Rc<std::cell::RefCell<#struct_name>>) -> Self {
                Self {
                    inner: std::rc::Rc::downgrade(value),
                }
            }

            pub fn upgrade(&self) -> Option<#shared_name> {
                self.inner.upgrade().map(#shared_name::from)
            }

            pub fn is_expired(&self) -> bool {
                self.inner.strong_count() == 0
            }

            pub fn get_weak(&self) -> &std::rc::Weak<std::cell::RefCell<#struct_name>> {
                &self.inner
            }

            #(#methods_weak)*
        }

        impl Clone for #weak_name {
            fn clone(&self) -> Self {
                Self {
                    inner: self.inner.clone(),
                }
            }
        }

        impl std::fmt::Debug for #weak_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!(#weak_name))
                    .field("is_expired", &self.is_expired())
                    .finish()
            }
        }

        impl From<std::rc::Weak<std::cell::RefCell<#struct_name>>> for #weak_name {
            fn from(inner: std::rc::Weak<std::cell::RefCell<#struct_name>>) -> Self {
                Self { inner }
            }
        }

        impl From<#weak_name> for std::rc::Weak<std::cell::RefCell<#struct_name>> {
            fn from(wrapper: #weak_name) -> Self {
                wrapper.inner
            }
        }
    };

    TokenStream::from(expanded)
}

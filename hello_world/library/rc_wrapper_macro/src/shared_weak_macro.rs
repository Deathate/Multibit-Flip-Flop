use quote::{format_ident, quote};
use syn::{DeriveInput, parse_macro_input};

pub fn expand(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let struct_name = &input.ident;
    let vis = &input.vis;
    let attrs = &input.attrs;
    let shared_name = format_ident!("Shared{}", struct_name);
    let weak_name = format_ident!("Weak{}", struct_name);
    let expanded = quote! {
        #(#attrs)*
        #vis struct #shared_name {
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

        #vis struct #weak_name {
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

    proc_macro::TokenStream::from(expanded)
}

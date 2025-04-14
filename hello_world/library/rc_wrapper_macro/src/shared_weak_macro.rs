use quote::{format_ident, quote};
use std::collections::HashSet;
use syn::{DeriveInput, Fields, parse_macro_input};

fn is_primitive_copy(ty: &syn::Type) -> bool {
    use syn::{Type, TypePath};

    let primitives: HashSet<&str> = [
        "u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128", "usize", "isize",
        "bool", "char", "f32", "f64", "float", "int", "uint"
    ]
    .iter()
    .cloned()
    .collect();

    if let Type::Path(TypePath { qself: None, path }) = ty {
        if let Some(ident) = path.get_ident() {
            return primitives.contains(ident.to_string().as_str());
        }
    }

    false
}

pub fn expand(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let struct_name = &input.ident;
    let vis = &input.vis;
    let attrs = &input.attrs;
    let shared_name = format_ident!("Shared{}", struct_name);
    let weak_name = format_ident!("Weak{}", struct_name);

    let fields = if let syn::Data::Struct(data) = &input.data {
        match &data.fields {
            Fields::Named(named) => named
                .named
                .iter()
                .filter(|f| {
                    matches!(f.vis, syn::Visibility::Public(_))
                        && !f.attrs.iter().any(|attr| attr.path().is_ident("skip"))
                })
                .collect::<Vec<_>>(),
            _ => vec![],
        }
    } else {
        vec![]
    };

    // Generate getter and setter methods
    let accessors = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;
        let getter = format_ident!("get_{}", name.as_ref().unwrap());
        let setter = format_ident!("set_{}", name.as_ref().unwrap());
        let getter_fn = if is_primitive_copy(&ty) {
            quote! {
                #[inline(always)]
                pub fn #getter(&self) -> #ty {
                    self.borrow().#name
                }
            }
        } else {
            quote! {
                #[inline(always)]
                pub fn #getter(&self) -> std::cell::Ref<#ty> {
                    std::cell::Ref::map(self.borrow(), |inner| &inner.#name)
                }
            }
        };
        quote! {
            #getter_fn
            #[inline(always)]
            pub fn #setter(&self, value: #ty) -> &Self {
                self.borrow_mut().#name = value;
                self
            }
        }
    });
    let accessors_weak = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;
        let setter = format_ident!("set_{}", name.as_ref().unwrap());

        quote! {
            #[inline(always)]
            pub fn #setter(&self, value: #ty) -> &Self {
                self.0.upgrade().unwrap().borrow_mut().#name = value;
                self
            }
        }
    });

    let expanded = quote! {
        #(#attrs)*
        #vis struct #shared_name(pub std::rc::Rc<std::cell::RefCell<#struct_name>>);

        impl #shared_name {
            pub fn new(value: #struct_name) -> Self {
                Self(std::rc::Rc::new(std::cell::RefCell::new(value)))
            }
            pub fn borrow(&self) -> std::cell::Ref<#struct_name> {
                self.0.borrow()
            }
            pub fn borrow_mut(&self) -> std::cell::RefMut<#struct_name> {
                self.0.borrow_mut()
            }
            pub fn get_ref(&self) -> &std::rc::Rc<std::cell::RefCell<#struct_name>> {
                &self.0
            }
            pub fn downgrade(&self) -> #weak_name {
                #weak_name(std::rc::Rc::downgrade(&self.0))
            }
            #(#accessors)*
        }

        impl Clone for #shared_name {
            fn clone(&self) -> Self {
                Self(self.0.clone())
            }
        }

        impl std::fmt::Debug for #shared_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!(#shared_name))
                    .field("inner", &self.0.borrow())
                    .finish()
            }
        }

        impl From<std::rc::Rc<std::cell::RefCell<#struct_name>>> for #shared_name {
            fn from(inner: std::rc::Rc<std::cell::RefCell<#struct_name>>) -> Self {
                Self(inner)
            }
        }

        impl From<#shared_name> for std::rc::Rc<std::cell::RefCell<#struct_name>> {
            fn from(wrapper: #shared_name) -> Self {
                wrapper.0
            }
        }

        #vis struct #weak_name(pub std::rc::Weak<std::cell::RefCell<#struct_name>>);

        impl #weak_name {
            pub fn new(value: &std::rc::Rc<std::cell::RefCell<#struct_name>>) -> Self {
                Self(std::rc::Rc::downgrade(value))
            }
            pub fn upgrade(&self) -> Option<#shared_name> {
                self.0.upgrade().map(#shared_name::from)
            }
            pub fn is_expired(&self) -> bool {
                self.0.strong_count() == 0
            }
            // pub fn get_ref(&self) -> &std::rc::Weak<std::cell::RefCell<#struct_name>> {
            //     &self.0
            // }
            #(#accessors_weak)*
        }

        impl Clone for #weak_name {
            fn clone(&self) -> Self {
                Self(self.0.clone())
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
                Self(inner)
            }
        }

        impl From<#weak_name> for std::rc::Weak<std::cell::RefCell<#struct_name>> {
            fn from(wrapper: #weak_name) -> Self {
                wrapper.0
            }
        }
    };

    proc_macro::TokenStream::from(expanded)
}

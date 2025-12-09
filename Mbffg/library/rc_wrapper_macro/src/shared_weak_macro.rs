use quote::{format_ident, quote};
use std::collections::HashSet;
use syn::{DeriveInput, Fields, parse_macro_input};

fn is_primitive_copy(ty: &syn::Type) -> bool {
    use syn::{Type, TypePath};

    let primitives: HashSet<&str> = [
        "u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128", "usize", "isize",
        "bool", "char", "f32", "f64", "float", "int", "uint",
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

    // 1. Collect all PUBLIC fields.
    // We removed the generic "skip" filter here because we need to check inside specific options.
    let fields = if let syn::Data::Struct(data) = &input.data {
        match &data.fields {
            Fields::Named(named) => named
                .named
                .iter()
                .filter(|f| matches!(f.vis, syn::Visibility::Public(_)))
                .collect::<Vec<_>>(),
            _ => vec![],
        }
    } else {
        vec![]
    };

    // Helper function to parse skip attributes
    let get_skip_flags = |attrs: &[syn::Attribute]| -> (bool, bool) {
        let mut skip_get = false;
        let mut skip_set = false;

        for attr in attrs {
            if attr.path().is_ident("skip") {
                match &attr.meta {
                    // Case: #[skip] -> Skip everything
                    syn::Meta::Path(_) => {
                        skip_get = true;
                        skip_set = true;
                    }
                    // Case: #[skip(get, set)] or #[skip(get)]
                    syn::Meta::List(list) => {
                        let _ = list.parse_nested_meta(|meta| {
                            if meta.path.is_ident("get") {
                                skip_get = true;
                            } else if meta.path.is_ident("set") {
                                skip_set = true;
                            }
                            Ok(())
                        });
                    }
                    _ => {}
                }
            }
        }
        (skip_get, skip_set)
    };

    // Generate getter and setter methods for Shared struct
    let accessors = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;
        let (skip_get, skip_set) = get_skip_flags(&f.attrs);

        let getter_block = if !skip_get {
            let getter = format_ident!("get_{}", name.as_ref().unwrap());
            let getter_mut = format_ident!("get_{}_mut", name.as_ref().unwrap());

            let getter_fn = if is_primitive_copy(&ty) {
                quote! {
                    pub fn #getter(&self) -> #ty {
                        self.borrow().#name
                    }
                }
            } else {
                quote! {
                    pub fn #getter(&self) -> std::cell::Ref<#ty> {
                        std::cell::Ref::map(self.borrow(), |inner| &inner.#name)
                    }
                }
            };

            quote! {
                #getter_fn
                pub fn #getter_mut(&self) -> std::cell::RefMut<#ty> {
                    std::cell::RefMut::map(self.borrow_mut(), |inner| &mut inner.#name)
                }
            }
        } else {
            quote! {}
        };

        let setter_block = if !skip_set {
            let setter = format_ident!("set_{}", name.as_ref().unwrap());
            quote! {
                pub fn #setter(&self, value: #ty) -> &Self {
                    self.borrow_mut().#name = value;
                    self
                }
            }
        } else {
            quote! {}
        };

        quote! {
            #getter_block
            #setter_block
        }
    });

    // Generate methods for Weak struct
    let accessors_weak = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;
        let (skip_get, skip_set) = get_skip_flags(&f.attrs);

        let getter_block = if !skip_get {
            let getter = format_ident!("get_{}", name.as_ref().unwrap());
            if is_primitive_copy(&ty) {
                quote! {
                    pub fn #getter(&self) -> #ty {
                        self.0.upgrade().unwrap().borrow().#name
                    }
                }
            } else {
                quote! {}
            }
        } else {
            quote! {}
        };

        let setter_block = if !skip_set {
            let setter = format_ident!("set_{}", name.as_ref().unwrap());
            quote! {
                pub fn #setter(&self, value: #ty) -> &Self {
                    self.0.upgrade().unwrap().borrow_mut().#name = value;
                    self
                }
            }
        } else {
            quote! {}
        };

        quote! {
            #getter_block
            #setter_block
        }
    });

    // --- Collect #[hash] fields ---
    let hash_fields: Vec<_> = if let syn::Data::Struct(data) = &input.data {
        match &data.fields {
            Fields::Named(named) => named
                .named
                .iter()
                .filter(|f| f.attrs.iter().any(|attr| attr.path().is_ident("hash")))
                .collect::<Vec<_>>(),
            _ => vec![],
        }
    } else {
        vec![]
    };

    let hash_idents: Vec<_> = hash_fields
        .iter()
        .map(|f| f.ident.as_ref().unwrap())
        .collect();

    let impl_hash = if !hash_idents.is_empty() {
        let eq_fields = hash_idents.iter();
        let hash_fields = hash_idents.iter();
        quote! {
            impl std::cmp::PartialEq for #struct_name {
                fn eq(&self, other: &Self) -> bool {
                    true #(&& self.#eq_fields == other.#eq_fields)*
                }
            }
            impl std::cmp::Eq for #struct_name {}
            impl std::hash::Hash for #struct_name {
                fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                    #(self.#hash_fields.hash(state);)*
                }
            }
        }
    } else {
        quote! {}
    };

    let impl_hash_shared = if !hash_idents.is_empty() {
        quote! {
            impl std::cmp::PartialEq for #shared_name {
                fn eq(&self, other: &Self) -> bool {
                    *self.0.borrow() == *other.0.borrow()
                }
            }
            impl std::cmp::Eq for #shared_name {}

            impl std::hash::Hash for #shared_name {
                fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                    // Use Rc's pointer address as the basis for hash
                    (&*self.0 as *const _ as usize).hash(state);
                }
            }
        }
    } else {
        quote! {}
    };

    let expanded = quote! {
        #impl_hash

        #(#attrs)*
        #vis struct #shared_name(pub std::rc::Rc<std::cell::RefCell<#struct_name>>);

        impl #shared_name {
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

        impl From<#struct_name> for #shared_name {
            fn from(inner: #struct_name) -> Self {
                Self(std::rc::Rc::new(std::cell::RefCell::new(inner)))
            }
        }

        impl From<#shared_name> for std::rc::Rc<std::cell::RefCell<#struct_name>> {
            fn from(wrapper: #shared_name) -> Self {
                wrapper.0
            }
        }

        #impl_hash_shared

        #vis struct #weak_name(pub std::rc::Weak<std::cell::RefCell<#struct_name>>);

        impl #weak_name {
            pub fn new(value: &std::rc::Rc<std::cell::RefCell<#struct_name>>) -> Self {
                Self(std::rc::Rc::downgrade(value))
            }
            pub fn upgrade(&self) -> Option<#shared_name> {
                self.0.upgrade().map(#shared_name::from)
            }
            pub fn upgrade_expect(&self) -> #shared_name {
                self.upgrade().expect("Failed to upgrade Weak reference")
            }
            pub fn is_expired(&self) -> bool {
                self.0.strong_count() == 0
            }
            #(#accessors_weak)*
        }

        impl Default for #weak_name {
            fn default() -> Self {
                Self(std::rc::Weak::new())
            }
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

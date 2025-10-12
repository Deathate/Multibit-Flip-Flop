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
        let getter_mut = format_ident!("get_{}_mut", name.as_ref().unwrap());
        let setter = format_ident!("set_{}", name.as_ref().unwrap());

        // For Copy primitives, return by value via a read lock
        let getter_fn = if is_primitive_copy(&ty) {
            quote! {
                pub fn #getter(&self) -> #ty {
                    self.borrow().#name
                }
            }
        } else {
            // For non-Copy, return a mapped read guard into the field
            // parking_lot: RwLockReadGuard::map -> MappedRwLockReadGuard<'_, FieldTy>
            quote! {
                pub fn #getter(&self) -> parking_lot::MappedRwLockReadGuard<'_, #ty> {
                    parking_lot::RwLockReadGuard::map(self.borrow_guard(), |inner| &inner.#name)
                }
            }
        };

        quote! {
            #getter_fn

            pub fn #getter_mut(&self) -> parking_lot::MappedRwLockWriteGuard<'_, #ty> {
                parking_lot::RwLockWriteGuard::map(self.borrow_write_guard(), |inner| &mut inner.#name)
            }

            pub fn #setter(&self, value: #ty) -> &Self {
                self.borrow_mut().#name = value;
                self
            }
        }
    });

    let accessors_weak = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;
        let getter = format_ident!("get_{}", name.as_ref().unwrap());
        let setter = format_ident!("set_{}", name.as_ref().unwrap());

        let getter_fn = if is_primitive_copy(&ty) {
            quote! {
                pub fn #getter(&self) -> #ty {
                    self.0.upgrade().unwrap().read().#name
                }
            }
        } else {
            // For non-Copy via Weak, we avoid returning guards referencing a temporary lock.
            quote! {}
        };

        quote! {
            #getter_fn

            pub fn #setter(&self, value: #ty) -> &Self {
                self.0.upgrade().unwrap().write().#name = value;
                self
            }
        }
    });

    // --- 1. Collect #[hash] fields ---
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
        let hash_fields_eq = hash_idents.iter();
        quote! {
            impl std::cmp::PartialEq for #struct_name {
                fn eq(&self, other: &Self) -> bool {
                    true #(&& self.#eq_fields == other.#eq_fields)*
                }
            }
            impl std::cmp::Eq for #struct_name {}

            impl std::hash::Hash for #struct_name {
                fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                    #(self.#hash_fields_eq.hash(state);)*
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
                    *self.0.read() == *other.0.read()
                }
            }
            impl std::cmp::Eq for #shared_name {}

            impl std::hash::Hash for #shared_name {
                fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                    // Hash by pointer (identity), fast and stable across the lifetime
                    (std::sync::Arc::as_ptr(&self.0) as usize).hash(state);
                }
            }
        }
    } else {
        quote! {}
    };

    let expanded = quote! {
        #impl_hash

        #(#attrs)*
        #vis struct #shared_name(pub std::sync::Arc<parking_lot::RwLock<#struct_name>>);

        impl #shared_name {
            pub fn new(value: #struct_name) -> Self {
                Self(std::sync::Arc::new(parking_lot::RwLock::new(value)))
            }

            /// Convenience: immutable access to the whole value (copy fields or clone by hand).
            pub fn borrow(&self) -> parking_lot::RwLockReadGuard<'_, #struct_name> {
                self.0.read()
            }

            /// Convenience: mutable access to the whole value.
            pub fn borrow_mut(&self) -> parking_lot::RwLockWriteGuard<'_, #struct_name> {
                self.0.write()
            }

            /// Internals for mapping; kept separate to avoid name collisions.
            fn borrow_guard(&self) -> parking_lot::RwLockReadGuard<'_, #struct_name> {
                self.0.read()
            }
            fn borrow_write_guard(&self) -> parking_lot::RwLockWriteGuard<'_, #struct_name> {
                self.0.write()
            }

            pub fn get_ref(&self) -> &std::sync::Arc<parking_lot::RwLock<#struct_name>> {
                &self.0
            }

            pub fn downgrade(&self) -> #weak_name {
                #weak_name(std::sync::Arc::downgrade(&self.0))
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
                    .field("inner", &"<Arc<parking_lot::RwLock<...>>>")
                    .finish()
            }
        }

        impl From<std::sync::Arc<parking_lot::RwLock<#struct_name>>> for #shared_name {
            fn from(inner: std::sync::Arc<parking_lot::RwLock<#struct_name>>) -> Self {
                Self(inner)
            }
        }

        impl From<#struct_name> for #shared_name {
            fn from(inner: #struct_name) -> Self {
                Self(std::sync::Arc::new(parking_lot::RwLock::new(inner)))
            }
        }

        impl From<#shared_name> for std::sync::Arc<parking_lot::RwLock<#struct_name>> {
            fn from(wrapper: #shared_name) -> Self {
                wrapper.0
            }
        }

        #impl_hash_shared

        #vis struct #weak_name(pub std::sync::Weak<parking_lot::RwLock<#struct_name>>);

        impl #weak_name {
            pub fn new(value: &std::sync::Arc<parking_lot::RwLock<#struct_name>>) -> Self {
                Self(std::sync::Arc::downgrade(value))
            }

            pub fn upgrade(&self) -> Option<#shared_name> {
                self.0.upgrade().map(#shared_name::from)
            }

            pub fn is_expired(&self) -> bool {
                self.0.strong_count() == 0
            }

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

        impl From<std::sync::Weak<parking_lot::RwLock<#struct_name>>> for #weak_name {
            fn from(inner: std::sync::Weak<parking_lot::RwLock<#struct_name>>) -> Self {
                Self(inner)
            }
        }

        impl From<#weak_name> for std::sync::Weak<parking_lot::RwLock<#struct_name>> {
            fn from(wrapper: #weak_name) -> Self {
                wrapper.0
            }
        }
    };

    proc_macro::TokenStream::from(expanded)
}

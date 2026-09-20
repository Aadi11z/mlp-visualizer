pub mod mlp {
    pub mod activations {
        pub use mlp_core::Activation;
    }

    pub mod network {
        pub use mlp_core::{
            ForwardTrace as ForwardResult, Gradients, LossKind, Network, NetworkConfig,
            NetworkConfigError, NetworkLimits,
        };
    }
}

pub mod engine {
    pub use mlp_core::*;
}

pub mod session;

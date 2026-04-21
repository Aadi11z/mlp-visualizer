// -----------------
// FILE: tests/mlp_tests.rs
// -----------------
// A simple unit test showing XOR learning might be included, but for brevity we'll show a tiny test

/*
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlp::activations::Activation;

    #[test]
    fn forward_identity() {
        let cfg = NetworkConfig { layer_sizes: vec![2,2], activations: vec![Activation::Identity] };
        let net = Network::from_config(cfg);
        let res = net.forward(&vec![1.0, 2.0]);
        assert_eq!(res.outputs.len(), 2);
    }
}
*/
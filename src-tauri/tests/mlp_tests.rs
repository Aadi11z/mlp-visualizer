use mlp_server::mlp::{
    activations::Activation,
    network::{LossKind, Network, NetworkConfig},
};

#[test]
fn native_server_facade_reexports_the_shared_network_engine() {
    let config = NetworkConfig::new(vec![2, 3, 1], vec![Activation::Tanh, Activation::Sigmoid])
        .expect("valid configuration");
    let network = Network::new(config, 42);
    let trace = network.forward(&[0.25, 0.75]).expect("forward pass");

    assert_eq!(trace.activations.len(), 3);
    assert_eq!(trace.outputs.len(), 1);
    assert!((0.0..=1.0).contains(&trace.outputs[0]));
}

#[test]
fn native_server_facade_can_train_with_shared_core_loss_contract() {
    let config = NetworkConfig::new(vec![2, 1], vec![Activation::Sigmoid]).unwrap();
    let mut network = Network::new(config, 42);
    let before = network.weights().to_vec();
    let sample = network
        .train_sample(&[0.0, 1.0], &[1.0], LossKind::MeanSquaredError, 0.1)
        .unwrap();

    assert!(sample.loss.is_finite());
    assert_ne!(network.weights(), before);
}

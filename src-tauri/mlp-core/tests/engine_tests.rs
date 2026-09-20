use mlp_core::{
    Activation, DatasetKind, DatasetSpec, EngineError, Experiment, ExperimentSpec,
    FeatureSelection, ForwardTrace, Gradients, InputFeature, LossKind, Network, NetworkConfig,
    NetworkConfigError, NetworkLimits,
};

fn xor_spec() -> ExperimentSpec {
    ExperimentSpec {
        layer_sizes: vec![2, 4, 1],
        activations: vec![Activation::Tanh, Activation::Sigmoid],
        features: vec![InputFeature::X1, InputFeature::X2],
        dataset: DatasetSpec::logical(DatasetKind::Xor),
        loss: LossKind::BinaryCrossEntropy,
        learning_rate: 0.1,
        batch_size: 1,
        shuffle: true,
        seed: 42,
    }
}

fn objective(network: &Network, inputs: &[f64], target: f64) -> f64 {
    let output = network.forward(inputs).unwrap().outputs[0];
    (output - target).powi(2)
}

#[test]
fn browser_policy_accepts_the_initial_playground_maximum() {
    let layers = vec![7, 8, 8, 8, 8, 8, 8, 1];
    let config = NetworkConfig::with_limits(
        layers.clone(),
        vec![Activation::Tanh; layers.len() - 1],
        NetworkLimits::browser(),
    )
    .unwrap();

    assert_eq!(config.parameter_count(), 433);
    assert_eq!(config.layer_sizes(), &[7, 8, 8, 8, 8, 8, 8, 1]);
}

#[test]
fn browser_policy_bounds_input_hidden_and_output_widths_independently() {
    for (layers, layer) in [(vec![8, 1], 0), (vec![2, 9, 1], 1), (vec![2, 2], 1)] {
        let activations = vec![Activation::Sigmoid; layers.len() - 1];
        assert!(matches!(
            NetworkConfig::with_limits(layers, activations, NetworkLimits::browser()),
            Err(NetworkConfigError::LayerTooWide { layer: actual, .. }) if actual == layer
        ));
    }
}

#[test]
fn network_config_rejects_invalid_shapes_and_overflow() {
    assert_eq!(
        NetworkConfig::new(vec![2], vec![]).unwrap_err(),
        NetworkConfigError::TooFewLayers
    );
    assert!(matches!(
        NetworkConfig::new(vec![2, 0, 1], vec![Activation::Tanh; 2]),
        Err(NetworkConfigError::ZeroWidth { layer: 1 })
    ));
    assert_eq!(
        NetworkConfig::with_limits(
            vec![usize::MAX, usize::MAX],
            vec![Activation::Tanh],
            NetworkLimits {
                max_layers: 2,
                max_input_width: usize::MAX,
                max_hidden_width: usize::MAX,
                max_output_width: usize::MAX,
                max_parameters: usize::MAX,
            },
        )
        .unwrap_err(),
        NetworkConfigError::ParameterCountOverflow
    );
}

#[test]
fn initialization_is_seeded_and_parameter_shapes_are_checked() {
    let config =
        NetworkConfig::new(vec![2, 3, 1], vec![Activation::Tanh, Activation::Sigmoid]).unwrap();
    let first = Network::new(config.clone(), 91);
    let second = Network::new(config.clone(), 91);
    assert_eq!(first.weights(), second.weights());
    assert_eq!(first.biases(), second.biases());

    let wide_config = NetworkConfig::new(
        vec![7, 8, 8, 1],
        vec![Activation::ReLU, Activation::Tanh, Activation::Sigmoid],
    )
    .unwrap();
    let wide_first = Network::new(wide_config.clone(), 91);
    let wide_second = Network::new(wide_config.clone(), 91);
    assert_eq!(wide_first.weights(), wide_second.weights());
    assert_eq!(wide_first.biases(), wide_second.biases());
    assert_eq!(wide_first.weights()[0].len(), 8);
    assert_eq!(wide_first.weights()[0][0].len(), 7);

    assert!(Network::from_parameters(config, 91, vec![vec![vec![1.0]]], vec![vec![0.0]]).is_err());
}

#[test]
fn activation_values_and_derivatives_match_finite_differences() {
    let cases = [
        (Activation::Identity, 0.37),
        (Activation::Sigmoid, 0.37),
        (Activation::ReLU, 0.37),
        (Activation::ReLU, -0.37),
        (Activation::Tanh, 0.37),
        (Activation::Step, 0.37),
        (Activation::Step, -0.37),
    ];
    let epsilon = 1e-6;
    for (activation, input) in cases {
        let numeric = (activation.apply(input + epsilon) - activation.apply(input - epsilon))
            / (2.0 * epsilon);
        let analytic = activation.derivative(input);
        assert!(
            (numeric - analytic).abs() < 1e-6,
            "{activation:?}: numeric={numeric}, analytic={analytic}"
        );
    }
    assert_eq!(Activation::ReLU.derivative(0.0), 0.0);
    assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 1e-15);
}

#[test]
fn mse_and_binary_cross_entropy_match_hand_calculated_values_and_gradients() {
    let config = NetworkConfig::new(vec![1, 1], vec![Activation::Sigmoid]).unwrap();
    let network =
        Network::from_parameters(config, 42, vec![vec![vec![0.0]]], vec![vec![0.0]]).unwrap();
    let input = [1.0];

    let (mse, _, mse_gradients) = network
        .loss_and_gradients(&input, &[1.0], LossKind::MeanSquaredError)
        .unwrap();
    assert!((mse - 0.25).abs() < 1e-15);
    assert!((mse_gradients.bias_gradients[0][0] + 0.25).abs() < 1e-15);
    assert!((mse_gradients.weight_gradients[0][0][0] + 0.25).abs() < 1e-15);

    let (bce, _, bce_gradients) = network
        .loss_and_gradients(&input, &[1.0], LossKind::BinaryCrossEntropy)
        .unwrap();
    assert!((bce - std::f64::consts::LN_2).abs() < 1e-12);
    assert!((bce_gradients.bias_gradients[0][0] + 0.5).abs() < 1e-15);
    assert!((bce_gradients.weight_gradients[0][0][0] + 0.5).abs() < 1e-15);
}

#[test]
fn fabricated_traces_and_gradient_shapes_are_rejected_without_mutation() {
    let config = NetworkConfig::new(vec![2, 1], vec![Activation::Sigmoid]).unwrap();
    let mut network = Network::new(config, 7);
    let before = network.weights().to_vec();
    let invalid_trace = ForwardTrace {
        input: vec![1.0],
        pre_activations: vec![],
        activations: vec![],
        outputs: vec![],
    };
    assert!(network
        .gradients_from_trace(&invalid_trace, &[1.0], LossKind::MeanSquaredError)
        .is_err());

    let invalid_gradients = Gradients {
        weight_gradients: vec![],
        bias_gradients: vec![],
        deltas: vec![],
        norm: 0.0,
    };
    assert!(network.apply_sgd(&invalid_gradients, 0.1).is_err());
    assert_eq!(network.weights(), before);
}

#[test]
fn feature_selection_rejects_empty_duplicate_and_non_finite_transforms() {
    assert!(FeatureSelection::new(vec![]).is_err());
    assert!(FeatureSelection::new(vec![InputFeature::X1, InputFeature::X1]).is_err());
    let square = FeatureSelection::new(vec![InputFeature::X1Squared]).unwrap();
    assert!(matches!(
        square.transform(&[1.0e308, 0.0]),
        Err(EngineError::NumericalFailure(_))
    ));
}

#[test]
fn backprop_gradient_matches_a_finite_difference() {
    let config =
        NetworkConfig::new(vec![2, 2, 1], vec![Activation::Tanh, Activation::Sigmoid]).unwrap();
    let weights = vec![vec![vec![0.2, -0.3], vec![0.4, 0.1]], vec![vec![0.5, -0.2]]];
    let biases = vec![vec![0.05, -0.1], vec![0.2]];
    let network =
        Network::from_parameters(config.clone(), 7, weights.clone(), biases.clone()).unwrap();
    let input = [0.3, -0.7];
    let target = 0.8;
    let (_, _, gradients) = network
        .loss_and_gradients(&input, &[target], LossKind::MeanSquaredError)
        .unwrap();

    let epsilon = 1e-6;
    let mut plus = weights.clone();
    plus[0][0][0] += epsilon;
    let mut minus = weights;
    minus[0][0][0] -= epsilon;
    let positive = Network::from_parameters(config.clone(), 7, plus, biases.clone()).unwrap();
    let negative = Network::from_parameters(config, 7, minus, biases).unwrap();
    let numeric = (objective(&positive, &input, target) - objective(&negative, &input, target))
        / (2.0 * epsilon);
    let analytic = gradients.weight_gradients[0][0][0];

    assert!(
        (numeric - analytic).abs() < 1e-6,
        "numeric={numeric} analytic={analytic}"
    );
}

#[test]
fn experiment_training_is_identical_when_updates_are_chunked() {
    let mut uninterrupted = Experiment::new(xor_spec(), NetworkLimits::browser()).unwrap();
    let mut chunked = Experiment::new(xor_spec(), NetworkLimits::browser()).unwrap();
    uninterrupted.train_updates(80).unwrap();
    for _ in 0..8 {
        chunked.train_updates(10).unwrap();
    }

    let full = uninterrupted.snapshot();
    let split = chunked.snapshot();
    assert_eq!(full.weights, split.weights);
    assert_eq!(full.biases, split.biases);
    assert_eq!(full.update_count, split.update_count);
    assert_eq!(full.samples_processed, split.samples_processed);
    assert_eq!(full.epoch, split.epoch);
    assert_eq!(full.revision, split.revision);
    assert_eq!(full.metrics_history.len(), split.metrics_history.len());
}

#[test]
fn selected_features_are_shared_by_inference_and_boundary_and_do_not_mutate_state() {
    let mut spec = xor_spec();
    spec.features = vec![InputFeature::X1Squared, InputFeature::SinX2];
    spec.layer_sizes[0] = 2;
    let experiment = Experiment::new(spec, NetworkLimits::browser()).unwrap();
    let before = experiment.snapshot();
    let boundary = experiment.decision_boundary(5).unwrap();
    let raw_points = boundary.iter().map(|p| vec![p.x, p.y]).collect::<Vec<_>>();
    let outputs = experiment.forward_batch(&raw_points).unwrap();

    for (point, output) in boundary.iter().zip(outputs) {
        assert!((point.probability - output[0]).abs() < 1e-12);
    }
    let after = experiment.snapshot();
    assert_eq!(before.weights, after.weights);
    assert_eq!(before.update_count, after.update_count);
    assert_eq!(before.revision, after.revision);
}

#[test]
fn direct_input_to_output_experiment_trains_and_reset_repeats_seeded_parameters() {
    let mut spec = xor_spec();
    spec.layer_sizes = vec![2, 1];
    spec.activations = vec![Activation::Sigmoid];
    let mut experiment = Experiment::new(spec, NetworkLimits::browser()).unwrap();
    let initial = experiment.snapshot();
    assert_eq!(initial.layer_sizes, vec![2, 1]);

    experiment.train_updates(1).unwrap();
    experiment.reset(42).unwrap();
    let reset = experiment.snapshot();
    assert_eq!(reset.weights, initial.weights);
    assert_eq!(reset.biases, initial.biases);
    assert_eq!(reset.epoch, 0);
    assert_eq!(reset.update_count, 0);
    assert_eq!(reset.samples_processed, 0);
    assert_eq!(reset.metrics_history.len(), 1);
}

#[test]
fn invalid_reconfiguration_is_atomic_and_valid_change_resets_state() {
    let mut experiment = Experiment::new(xor_spec(), NetworkLimits::browser()).unwrap();
    experiment.train_updates(4).unwrap();
    let before = experiment.snapshot();

    let mut invalid = xor_spec();
    invalid.layer_sizes[0] = 1;
    assert!(experiment
        .reconfigure(invalid, NetworkLimits::browser())
        .is_err());
    let unchanged = experiment.snapshot();
    assert_eq!(before.weights, unchanged.weights);
    assert_eq!(before.update_count, unchanged.update_count);
    assert_eq!(before.revision, unchanged.revision);

    let mut changed = xor_spec();
    changed.seed = 99;
    experiment
        .reconfigure(changed, NetworkLimits::browser())
        .unwrap();
    let replacement = experiment.snapshot();
    assert_eq!(replacement.revision, before.revision + 1);
    assert_eq!(replacement.update_count, 0);
    assert_eq!(replacement.samples_processed, 0);
    assert_eq!(replacement.epoch, 0);
    assert_ne!(replacement.weights, before.weights);
}

#[test]
fn step_activation_and_invalid_binary_cross_entropy_contract_are_rejected() {
    let mut spec = xor_spec();
    spec.activations[0] = Activation::Step;
    assert!(matches!(
        Experiment::new(spec.clone(), NetworkLimits::browser()),
        Err(EngineError::UnsupportedConfiguration(_))
    ));

    spec.activations[0] = Activation::Tanh;
    spec.activations[1] = Activation::Identity;
    assert!(matches!(
        Experiment::new(spec, NetworkLimits::browser()),
        Err(EngineError::UnsupportedConfiguration(_))
    ));
}

#[test]
fn dataset_generation_and_resource_bounds_are_deterministic() {
    let spec = DatasetSpec::generated(DatasetKind::Spirals, 200, 0.1, 88);
    let a = mlp_core::RawDataset::generate(spec.clone()).unwrap();
    let b = mlp_core::RawDataset::generate(spec).unwrap();
    assert_eq!(a.examples, b.examples);
    assert_eq!(a.examples.len(), 200);
    assert_eq!(a.bounds, [-1.5, 1.5, -1.5, 1.5]);
    assert_eq!(
        a.examples
            .iter()
            .filter(|example| example.target == [0.0])
            .count(),
        100
    );
    assert_eq!(
        a.examples
            .iter()
            .filter(|example| example.target == [1.0])
            .count(),
        100
    );
    assert!(mlp_core::RawDataset::generate(DatasetSpec::generated(
        DatasetKind::Moons,
        199,
        0.1,
        1
    ))
    .is_err());
    assert!(mlp_core::RawDataset::generate(DatasetSpec::generated(
        DatasetKind::Moons,
        200,
        f64::NAN,
        1
    ))
    .is_err());

    let mut experiment = Experiment::new(xor_spec(), NetworkLimits::browser()).unwrap();
    assert!(experiment.decision_boundary(101).is_err());
    assert!(experiment
        .forward_batch(&vec![vec![0.0, 0.0]; 10_001])
        .is_err());
    assert!(experiment.train_updates(1_001).is_err());
}

#[test]
fn metrics_history_stays_bounded() {
    let mut experiment = Experiment::new(xor_spec(), NetworkLimits::browser()).unwrap();
    experiment.train_updates(1_000).unwrap();
    experiment.train_updates(1_000).unwrap();
    experiment.train_updates(100).unwrap();
    let snapshot = experiment.snapshot();
    assert_eq!(snapshot.metrics_history.len(), 512);
    assert!(snapshot.metrics_history[0].epoch > 0);
}

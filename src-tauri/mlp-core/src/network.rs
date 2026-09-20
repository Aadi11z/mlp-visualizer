use crate::{Activation, EngineError, NetworkConfigError};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct NetworkLimits {
    /// Includes input and output layers.
    pub max_layers: usize,
    pub max_input_width: usize,
    pub max_hidden_width: usize,
    pub max_output_width: usize,
    pub max_parameters: usize,
}

impl Default for NetworkLimits {
    fn default() -> Self {
        Self {
            max_layers: 32,
            max_input_width: 256,
            max_hidden_width: 256,
            max_output_width: 256,
            max_parameters: 1_000_000,
        }
    }
}

impl NetworkLimits {
    pub fn browser() -> Self {
        Self {
            max_layers: 8,
            max_input_width: 7,
            max_hidden_width: 8,
            max_output_width: 1,
            max_parameters: 10_000,
        }
    }

    /// Test-only-in-product benchmark envelope for measuring unexposed widths.
    pub fn measurement_candidates() -> Self {
        Self {
            max_layers: 8,
            max_input_width: 7,
            max_hidden_width: 32,
            max_output_width: 1,
            max_parameters: 10_000,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct NetworkConfig {
    layer_sizes: Vec<usize>,
    activations: Vec<Activation>,
    parameter_count: usize,
}

impl NetworkConfig {
    pub fn new(
        layer_sizes: Vec<usize>,
        activations: Vec<Activation>,
    ) -> Result<Self, NetworkConfigError> {
        Self::with_limits(layer_sizes, activations, NetworkLimits::default())
    }

    pub fn with_limits(
        layer_sizes: Vec<usize>,
        activations: Vec<Activation>,
        limits: NetworkLimits,
    ) -> Result<Self, NetworkConfigError> {
        if layer_sizes.len() < 2 {
            return Err(NetworkConfigError::TooFewLayers);
        }
        if layer_sizes.len() > limits.max_layers {
            return Err(NetworkConfigError::TooManyLayers {
                actual: layer_sizes.len(),
                maximum: limits.max_layers,
            });
        }
        let last_layer = layer_sizes.len() - 1;
        for (layer, &width) in layer_sizes.iter().enumerate() {
            if width == 0 {
                return Err(NetworkConfigError::ZeroWidth { layer });
            }
            let maximum = if layer == 0 {
                limits.max_input_width
            } else if layer == last_layer {
                limits.max_output_width
            } else {
                limits.max_hidden_width
            };
            if width > maximum {
                return Err(NetworkConfigError::LayerTooWide {
                    layer,
                    actual: width,
                    maximum,
                });
            }
        }
        let expected = layer_sizes.len() - 1;
        if activations.len() != expected {
            return Err(NetworkConfigError::ActivationCountMismatch {
                expected,
                actual: activations.len(),
            });
        }
        let parameter_count = layer_sizes.windows(2).try_fold(0usize, |sum, pair| {
            let layer_parameters = pair[0].checked_mul(pair[1])?.checked_add(pair[1])?;
            sum.checked_add(layer_parameters)
        });
        let parameter_count = parameter_count.ok_or(NetworkConfigError::ParameterCountOverflow)?;
        if parameter_count > limits.max_parameters {
            return Err(NetworkConfigError::TooManyParameters {
                actual: parameter_count,
                maximum: limits.max_parameters,
            });
        }
        Ok(Self {
            layer_sizes,
            activations,
            parameter_count,
        })
    }

    pub fn layer_sizes(&self) -> &[usize] {
        &self.layer_sizes
    }

    pub fn activations(&self) -> &[Activation] {
        &self.activations
    }

    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LossKind {
    MeanSquaredError,
    BinaryCrossEntropy,
}

impl LossKind {
    pub fn validate_for(&self, activations: &[Activation]) -> Result<(), EngineError> {
        if activations
            .iter()
            .any(|activation| !activation.is_differentiable())
        {
            return Err(EngineError::UnsupportedConfiguration(
                "step activation is inference-only and cannot be trained".into(),
            ));
        }
        if *self == Self::BinaryCrossEntropy && activations.last() != Some(&Activation::Sigmoid) {
            return Err(EngineError::UnsupportedConfiguration(
                "binary cross-entropy requires a sigmoid output activation".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn value(self, outputs: &[f64], targets: &[f64]) -> Result<f64, EngineError> {
        if outputs.len() != targets.len() || outputs.is_empty() {
            return Err(EngineError::ShapeMismatch(
                "prediction and target lengths must match and be non-empty".into(),
            ));
        }
        if targets.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidValue("targets must be finite".into()));
        }
        let count = outputs.len() as f64;
        let loss = match self {
            Self::MeanSquaredError => {
                outputs
                    .iter()
                    .zip(targets)
                    .map(|(output, target)| (output - target).powi(2))
                    .sum::<f64>()
                    / count
            }
            Self::BinaryCrossEntropy => {
                if targets.iter().any(|target| !(0.0..=1.0).contains(target)) {
                    return Err(EngineError::InvalidValue(
                        "binary cross-entropy targets must be in [0, 1]".into(),
                    ));
                }
                let epsilon = 1e-12;
                -outputs
                    .iter()
                    .zip(targets)
                    .map(|(output, target)| {
                        let probability = output.clamp(epsilon, 1.0 - epsilon);
                        target * probability.ln() + (1.0 - target) * (1.0 - probability).ln()
                    })
                    .sum::<f64>()
                    / count
            }
        };
        if loss.is_finite() {
            Ok(loss)
        } else {
            Err(EngineError::NumericalFailure("loss is not finite".into()))
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ForwardTrace {
    pub input: Vec<f64>,
    /// Contains one vector per layer, excluding the input layer.
    pub pre_activations: Vec<Vec<f64>>,
    /// Includes the input vector as element zero.
    pub activations: Vec<Vec<f64>>,
    pub outputs: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Gradients {
    pub weight_gradients: Vec<Vec<Vec<f64>>>,
    pub bias_gradients: Vec<Vec<f64>>,
    pub deltas: Vec<Vec<f64>>,
    pub norm: f64,
}

impl Gradients {
    fn zeros_for(network: &Network) -> Self {
        Self {
            weight_gradients: network
                .weights
                .iter()
                .map(|layer| layer.iter().map(|row| vec![0.0; row.len()]).collect())
                .collect(),
            bias_gradients: network
                .biases
                .iter()
                .map(|layer| vec![0.0; layer.len()])
                .collect(),
            deltas: network
                .biases
                .iter()
                .map(|layer| vec![0.0; layer.len()])
                .collect(),
            norm: 0.0,
        }
    }

    fn add_scaled(&mut self, other: &Self, scale: f64) {
        for (dst_layer, src_layer) in self
            .weight_gradients
            .iter_mut()
            .zip(&other.weight_gradients)
        {
            for (dst_row, src_row) in dst_layer.iter_mut().zip(src_layer) {
                for (dst, src) in dst_row.iter_mut().zip(src_row) {
                    *dst += src * scale;
                }
            }
        }
        for (dst_layer, src_layer) in self.bias_gradients.iter_mut().zip(&other.bias_gradients) {
            for (dst, src) in dst_layer.iter_mut().zip(src_layer) {
                *dst += src * scale;
            }
        }
    }

    fn finish_norm(&mut self) {
        let squared_norm = self
            .weight_gradients
            .iter()
            .flatten()
            .flatten()
            .chain(self.bias_gradients.iter().flatten())
            .map(|value| value * value)
            .sum::<f64>();
        self.norm = squared_norm.sqrt();
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SampleResult {
    pub loss: f64,
    pub outputs: Vec<f64>,
    pub gradient_norm: f64,
}

pub struct Network {
    config: NetworkConfig,
    seed: u64,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
}

impl Network {
    pub fn new(config: NetworkConfig, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut weights = Vec::with_capacity(config.layer_sizes.len() - 1);
        let mut biases = Vec::with_capacity(config.layer_sizes.len() - 1);
        for (index, pair) in config.layer_sizes.windows(2).enumerate() {
            let fan_in = pair[0];
            let fan_out = pair[1];
            let activation = config.activations[index];
            let scale = match activation {
                Activation::ReLU => (2.0 / fan_in as f64).sqrt(),
                _ => (2.0 / (fan_in + fan_out) as f64).sqrt(),
            };
            let mut layer = vec![vec![0.0; fan_in]; fan_out];
            for row in &mut layer {
                for weight in row {
                    let sample: f64 = StandardNormal.sample(&mut rng);
                    *weight = sample * scale;
                }
            }
            weights.push(layer);
            biases.push(vec![0.0; fan_out]);
        }
        Self {
            config,
            seed,
            weights,
            biases,
        }
    }

    pub fn from_parameters(
        config: NetworkConfig,
        seed: u64,
        weights: Vec<Vec<Vec<f64>>>,
        biases: Vec<Vec<f64>>,
    ) -> Result<Self, EngineError> {
        validate_parameters(&config, &weights, &biases)?;
        Ok(Self {
            config,
            seed,
            weights,
            biases,
        })
    }

    pub fn config(&self) -> &NetworkConfig {
        &self.config
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn weights(&self) -> &[Vec<Vec<f64>>] {
        &self.weights
    }
    pub fn biases(&self) -> &[Vec<f64>] {
        &self.biases
    }

    pub fn forward(&self, inputs: &[f64]) -> Result<ForwardTrace, EngineError> {
        let expected = self.config.layer_sizes[0];
        if inputs.len() != expected {
            return Err(EngineError::ShapeMismatch(format!(
                "expected {expected} inputs, received {}",
                inputs.len()
            )));
        }
        if inputs.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidValue("inputs must be finite".into()));
        }
        let mut current = inputs.to_vec();
        let mut activations = vec![current.clone()];
        let mut pre_activations = Vec::with_capacity(self.weights.len());
        for (layer_index, (weights, biases)) in self.weights.iter().zip(&self.biases).enumerate() {
            let mut z = Vec::with_capacity(weights.len());
            let mut next = Vec::with_capacity(weights.len());
            for (row, bias) in weights.iter().zip(biases) {
                let sum = row.iter().zip(&current).map(|(w, a)| w * a).sum::<f64>() + bias;
                let activation = self.config.activations[layer_index].apply(sum);
                if !(sum.is_finite() && activation.is_finite()) {
                    return Err(EngineError::NumericalFailure(
                        "forward pass produced a non-finite value".into(),
                    ));
                }
                z.push(sum);
                next.push(activation);
            }
            pre_activations.push(z);
            current = next;
            activations.push(current.clone());
        }
        Ok(ForwardTrace {
            input: inputs.to_vec(),
            pre_activations,
            activations,
            outputs: current,
        })
    }

    pub fn loss_and_gradients(
        &self,
        inputs: &[f64],
        targets: &[f64],
        loss: LossKind,
    ) -> Result<(f64, ForwardTrace, Gradients), EngineError> {
        loss.validate_for(&self.config.activations)?;
        let trace = self.forward(inputs)?;
        let value = loss.value(&trace.outputs, targets)?;
        let gradients = self.gradients_from_trace(&trace, targets, loss)?;
        Ok((value, trace, gradients))
    }

    pub fn train_sample(
        &mut self,
        inputs: &[f64],
        targets: &[f64],
        loss: LossKind,
        learning_rate: f64,
    ) -> Result<SampleResult, EngineError> {
        let (value, trace, gradients) = self.loss_and_gradients(inputs, targets, loss)?;
        let gradient_norm = gradients.norm;
        self.apply_sgd(&gradients, learning_rate)?;
        Ok(SampleResult {
            loss: value,
            outputs: trace.outputs,
            gradient_norm,
        })
    }

    pub fn gradients_from_trace(
        &self,
        trace: &ForwardTrace,
        targets: &[f64],
        loss: LossKind,
    ) -> Result<Gradients, EngineError> {
        loss.validate_for(&self.config.activations)?;
        let layer_count = self.weights.len();
        if trace.input.len() != self.config.layer_sizes[0]
            || trace.activations.len() != layer_count + 1
            || trace.pre_activations.len() != layer_count
            || trace.outputs.len() != *self.config.layer_sizes.last().unwrap()
        {
            return Err(EngineError::ShapeMismatch(
                "forward trace does not match network configuration".into(),
            ));
        }
        if trace
            .activations
            .iter()
            .enumerate()
            .any(|(layer, values)| values.len() != self.config.layer_sizes[layer])
            || trace
                .pre_activations
                .iter()
                .enumerate()
                .any(|(layer, values)| values.len() != self.config.layer_sizes[layer + 1])
            || trace.activations[0] != trace.input
            || trace.activations.last().unwrap() != &trace.outputs
            || trace
                .input
                .iter()
                .chain(trace.outputs.iter())
                .chain(trace.activations.iter().flatten())
                .chain(trace.pre_activations.iter().flatten())
                .any(|value| !value.is_finite())
        {
            return Err(EngineError::ShapeMismatch(
                "forward trace contains inconsistent or non-finite values".into(),
            ));
        }
        if targets.len() != self.config.layer_sizes[self.config.layer_sizes.len() - 1]
            || targets.iter().any(|value| !value.is_finite())
        {
            return Err(EngineError::ShapeMismatch(
                "targets do not match network output".into(),
            ));
        }
        let output_count = targets.len() as f64;
        let mut gradients = Gradients::zeros_for(self);
        for (output_index, target) in targets.iter().enumerate() {
            let output = trace.outputs[output_index];
            let final_activation = self.config.activations[layer_count - 1];
            let delta = if loss == LossKind::BinaryCrossEntropy {
                (output - target) / output_count
            } else {
                (2.0 * (output - target) / output_count)
                    * final_activation
                        .derivative(trace.pre_activations[layer_count - 1][output_index])
            };
            gradients.deltas[layer_count - 1][output_index] = delta;
        }
        for layer_index in (0..layer_count - 1).rev() {
            for neuron in 0..self.config.layer_sizes[layer_index + 1] {
                let downstream = self.weights[layer_index + 1]
                    .iter()
                    .zip(&gradients.deltas[layer_index + 1])
                    .map(|(row, delta)| row[neuron] * delta)
                    .sum::<f64>();
                gradients.deltas[layer_index][neuron] = downstream
                    * self.config.activations[layer_index]
                        .derivative(trace.pre_activations[layer_index][neuron]);
            }
        }
        for layer_index in 0..layer_count {
            let layer_input = &trace.activations[layer_index];
            for neuron in 0..self.weights[layer_index].len() {
                for (input_index, value) in layer_input.iter().enumerate() {
                    gradients.weight_gradients[layer_index][neuron][input_index] =
                        gradients.deltas[layer_index][neuron] * value;
                }
                gradients.bias_gradients[layer_index][neuron] =
                    gradients.deltas[layer_index][neuron];
            }
        }
        gradients.finish_norm();
        if !gradients.norm.is_finite() {
            return Err(EngineError::NumericalFailure(
                "gradient is not finite".into(),
            ));
        }
        Ok(gradients)
    }

    pub fn apply_sgd(
        &mut self,
        gradients: &Gradients,
        learning_rate: f64,
    ) -> Result<(), EngineError> {
        if !(learning_rate.is_finite() && learning_rate > 0.0 && learning_rate < 100.0) {
            return Err(EngineError::InvalidValue(
                "learning rate must be in (0, 100)".into(),
            ));
        }
        if gradients.weight_gradients.len() != self.weights.len()
            || gradients.bias_gradients.len() != self.biases.len()
        {
            return Err(EngineError::ShapeMismatch(
                "gradient parameter layers do not match network".into(),
            ));
        }
        for (layer_index, layer) in self.weights.iter().enumerate() {
            if gradients.weight_gradients[layer_index].len() != layer.len()
                || gradients.bias_gradients[layer_index].len() != self.biases[layer_index].len()
            {
                return Err(EngineError::ShapeMismatch(format!(
                    "gradient shape mismatch at layer {layer_index}"
                )));
            }
            for (neuron, row) in layer.iter().enumerate() {
                if gradients.weight_gradients[layer_index][neuron].len() != row.len() {
                    return Err(EngineError::ShapeMismatch(format!(
                        "gradient shape mismatch at layer {layer_index}, neuron {neuron}"
                    )));
                }
                for (input_index, value) in row.iter().enumerate() {
                    let updated = value
                        - learning_rate
                            * gradients.weight_gradients[layer_index][neuron][input_index];
                    if !updated.is_finite() {
                        return Err(EngineError::NumericalFailure(
                            "SGD update would be non-finite".into(),
                        ));
                    }
                }
                let updated_bias = self.biases[layer_index][neuron]
                    - learning_rate * gradients.bias_gradients[layer_index][neuron];
                if !updated_bias.is_finite() {
                    return Err(EngineError::NumericalFailure(
                        "SGD update would be non-finite".into(),
                    ));
                }
            }
        }
        for (layer_index, layer) in self.weights.iter_mut().enumerate() {
            for (neuron, row) in layer.iter_mut().enumerate() {
                for (input_index, value) in row.iter_mut().enumerate() {
                    *value -= learning_rate
                        * gradients.weight_gradients[layer_index][neuron][input_index];
                }
                self.biases[layer_index][neuron] -=
                    learning_rate * gradients.bias_gradients[layer_index][neuron];
            }
        }
        Ok(())
    }

    pub fn train_batch<'a>(
        &mut self,
        examples: impl IntoIterator<Item = &'a crate::Example>,
        example_count: usize,
        loss: LossKind,
        learning_rate: f64,
    ) -> Result<SampleResult, EngineError> {
        if example_count == 0 {
            return Err(EngineError::InvalidValue(
                "training batch cannot be empty".into(),
            ));
        }
        let mut aggregate = Gradients::zeros_for(self);
        let mut mean_loss = 0.0;
        let mut last_outputs = Vec::new();
        let mut processed = 0usize;
        for example in examples {
            let (value, trace, gradients) =
                self.loss_and_gradients(&example.inputs, &example.target, loss)?;
            mean_loss += value;
            aggregate.add_scaled(&gradients, 1.0 / example_count as f64);
            last_outputs = trace.outputs;
            processed += 1;
        }
        if processed != example_count {
            return Err(EngineError::ShapeMismatch(
                "training batch count does not match its examples".into(),
            ));
        }
        aggregate.finish_norm();
        let gradient_norm = aggregate.norm;
        self.apply_sgd(&aggregate, learning_rate)?;
        Ok(SampleResult {
            loss: mean_loss / example_count as f64,
            outputs: last_outputs,
            gradient_norm,
        })
    }
}

fn validate_parameters(
    config: &NetworkConfig,
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>],
) -> Result<(), EngineError> {
    let layers = config.layer_sizes.len() - 1;
    if weights.len() != layers || biases.len() != layers {
        return Err(EngineError::ShapeMismatch(
            "parameter layer count does not match config".into(),
        ));
    }
    for layer in 0..layers {
        let expected_in = config.layer_sizes[layer];
        let expected_out = config.layer_sizes[layer + 1];
        if weights[layer].len() != expected_out
            || weights[layer].iter().any(|row| row.len() != expected_in)
            || biases[layer].len() != expected_out
        {
            return Err(EngineError::ShapeMismatch(format!(
                "parameter shape mismatch at layer {layer}"
            )));
        }
        if weights[layer]
            .iter()
            .flatten()
            .chain(&biases[layer])
            .any(|value| !value.is_finite())
        {
            return Err(EngineError::InvalidValue(
                "parameters must be finite".into(),
            ));
        }
    }
    Ok(())
}

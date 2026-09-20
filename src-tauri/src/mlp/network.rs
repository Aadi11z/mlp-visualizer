use crate::mlp::activations::Activation;
use rand_distr::Distribution;
use rand_distr::StandardNormal;
use serde::Serialize;
use std::fmt;

/// Resource limits applied when constructing a network configuration.
/// `max_layers` includes the input and output layers.
#[derive(Debug, Clone, Copy)]
pub struct NetworkLimits {
    pub max_layers: usize,
    pub max_width: usize,
    pub max_parameters: usize,
}

impl Default for NetworkLimits {
    fn default() -> Self {
        Self {
            max_layers: 6,
            max_width: 16,
            max_parameters: 10_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkConfigError {
    TooFewLayers,
    TooManyLayers {
        actual: usize,
        maximum: usize,
    },
    ZeroWidth {
        layer: usize,
    },
    LayerTooWide {
        layer: usize,
        actual: usize,
        maximum: usize,
    },
    ActivationCountMismatch {
        expected: usize,
        actual: usize,
    },
    TooManyParameters {
        actual: usize,
        maximum: usize,
    },
}

impl fmt::Display for NetworkConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewLayers => write!(f, "a network must have input and output layers"),
            Self::TooManyLayers { actual, maximum } => {
                write!(f, "network has {actual} layers; maximum is {maximum}")
            }
            Self::ZeroWidth { layer } => {
                write!(f, "layer {layer} must contain at least one neuron")
            }
            Self::LayerTooWide {
                layer,
                actual,
                maximum,
            } => {
                write!(f, "layer {layer} has width {actual}; maximum is {maximum}")
            }
            Self::ActivationCountMismatch { expected, actual } => {
                write!(f, "expected {expected} activations, received {actual}")
            }
            Self::TooManyParameters { actual, maximum } => {
                write!(f, "network has {actual} parameters; maximum is {maximum}")
            }
        }
    }
}

impl std::error::Error for NetworkConfigError {}

#[derive(Debug, Clone, Serialize)]
pub struct NetworkConfig {
    layer_sizes: Vec<usize>,
    activations: Vec<Activation>,
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
        for (layer, &width) in layer_sizes.iter().enumerate() {
            if width == 0 {
                return Err(NetworkConfigError::ZeroWidth { layer });
            }
            if width > limits.max_width {
                return Err(NetworkConfigError::LayerTooWide {
                    layer,
                    actual: width,
                    maximum: limits.max_width,
                });
            }
        }
        let expected_activations = layer_sizes.len() - 1;
        if activations.len() != expected_activations {
            return Err(NetworkConfigError::ActivationCountMismatch {
                expected: expected_activations,
                actual: activations.len(),
            });
        }

        // Include biases in the resource estimate and use checked arithmetic so
        // untrusted dimensions cannot overflow the validation calculation.
        let parameter_count = layer_sizes
            .windows(2)
            .try_fold(0usize, |total, layer| {
                let layer_parameters = layer[0].checked_mul(layer[1])?.checked_add(layer[1])?;
                total.checked_add(layer_parameters)
            })
            .unwrap_or(usize::MAX);
        if parameter_count > limits.max_parameters {
            return Err(NetworkConfigError::TooManyParameters {
                actual: parameter_count,
                maximum: limits.max_parameters,
            });
        }

        Ok(Self {
            layer_sizes,
            activations,
        })
    }

    pub fn layer_sizes(&self) -> &[usize] {
        &self.layer_sizes
    }

    pub fn activations(&self) -> &[Activation] {
        &self.activations
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ForwardResult {
    pub outputs: Vec<f64>,
    pub layer_outputs: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainStepResult {
    pub outputs: Vec<f64>,
    pub loss: f64,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
}

pub struct Network {
    config: NetworkConfig,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
}

impl Network {
    /// Builds a network from a configuration that has already passed validation.
    pub fn from_config(cfg: NetworkConfig) -> Self {
        let mut rng = rand::rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..(cfg.layer_sizes.len() - 1) {
            let in_size = cfg.layer_sizes[i];
            let out_size = cfg.layer_sizes[i + 1];
            let mut layer_w = vec![vec![0.0; in_size]; out_size];
            let layer_b = vec![0.0; out_size];
            let scale = (1.0 / in_size as f64).sqrt();
            for o in 0..out_size {
                for inp in 0..in_size {
                    let v: f64 = StandardNormal.sample(&mut rng);
                    layer_w[o][inp] = v * scale;
                }
            }
            weights.push(layer_w);
            biases.push(layer_b);
        }

        Network {
            config: cfg,
            weights,
            biases,
        }
    }

    pub fn config(&self) -> &NetworkConfig {
        &self.config
    }

    pub fn forward(&self, inputs: &Vec<f64>) -> ForwardResult {
        let mut current = inputs.clone();
        let mut layer_outputs: Vec<Vec<f64>> = vec![current.clone()];

        for (layer_idx, (w_layer, b_layer)) in
            self.weights.iter().zip(self.biases.iter()).enumerate()
        {
            let mut next = vec![0.0; w_layer.len()];
            let activation = self.config.activations[layer_idx];
            for o in 0..w_layer.len() {
                let mut sum = b_layer[o];
                for inp in 0..current.len() {
                    sum += w_layer[o][inp] * current[inp];
                }
                next[o] = activation.apply(sum);
            }
            layer_outputs.push(next.clone());
            current = next;
        }

        ForwardResult {
            outputs: current,
            layer_outputs,
        }
    }

    pub fn train_step(
        &mut self,
        inputs: &Vec<f64>,
        target: &Vec<f64>,
        learning_rate: f64,
    ) -> TrainStepResult {
        let mut activations_cache: Vec<Vec<f64>> = Vec::new();
        let mut pre_activations: Vec<Vec<f64>> = Vec::new();

        let mut current = inputs.clone();
        activations_cache.push(current.clone());

        for (layer_idx, (w_layer, b_layer)) in
            self.weights.iter().zip(self.biases.iter()).enumerate()
        {
            let act_fn = self.config.activations[layer_idx];
            let mut z = vec![0.0; w_layer.len()];
            let mut a = vec![0.0; w_layer.len()];
            for o in 0..w_layer.len() {
                let mut sum = b_layer[o];
                for inp in 0..current.len() {
                    sum += w_layer[o][inp] * current[inp];
                }
                z[o] = sum;
                a[o] = act_fn.apply(sum);
            }
            pre_activations.push(z);
            activations_cache.push(a.clone());
            current = a;
        }

        let outputs = activations_cache.last().cloned().unwrap_or_default();

        let mut loss = 0.0;
        for i in 0..outputs.len() {
            let diff = outputs[i] - target[i];
            loss += diff * diff;
        }
        loss /= outputs.len() as f64;

        let mut delta: Vec<Vec<f64>> = vec![vec![]; self.weights.len()];
        let num_layers = self.weights.len();

        if num_layers == 0 {
            return TrainStepResult {
                outputs,
                loss,
                weights: self.weights.clone(),
                biases: self.biases.clone(),
            };
        }

        let mut last_delta = vec![0.0; self.weights[num_layers - 1].len()];
        let act_fn = self.config.activations[num_layers - 1];
        for i in 0..last_delta.len() {
            let a = activations_cache[num_layers][i];
            let z = pre_activations[num_layers - 1][i];
            let dloss_da = 2.0 * (a - target[i]) / (outputs.len() as f64);
            let da_dz = act_fn.derivative(z);
            last_delta[i] = dloss_da * da_dz;
        }
        delta[num_layers - 1] = last_delta;

        for l in (0..num_layers - 1).rev() {
            let mut layer_delta = vec![0.0; self.weights[l].len()];
            let act_fn = self.config.activations[l];
            for i in 0..self.weights[l].len() {
                let mut sum = 0.0;
                for j in 0..self.weights[l + 1].len() {
                    sum += self.weights[l + 1][j][i] * delta[l + 1][j];
                }
                let z = pre_activations[l][i];
                layer_delta[i] = sum * act_fn.derivative(z);
            }
            delta[l] = layer_delta;
        }

        for l in 0..num_layers {
            let inputs_to_use = &activations_cache[l];
            for j in 0..self.weights[l].len() {
                for i in 0..self.weights[l][j].len() {
                    let grad = delta[l][j] * inputs_to_use[i];
                    self.weights[l][j][i] -= learning_rate * grad;
                }
                self.biases[l][j] -= learning_rate * delta[l][j];
            }
        }

        TrainStepResult {
            outputs,
            loss,
            weights: self.weights.clone(),
            biases: self.biases.clone(),
        }
    }
}

use crate::{
    Activation, DatasetKind, DatasetSpec, EngineError, Example, FeatureSelection, InputFeature,
    LossKind, Network, NetworkConfig, NetworkLimits, RawDataset,
};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const MAX_UPDATES_PER_CHUNK: usize = 1_000;
const MAX_BATCH_SIZE: usize = 512;
const MAX_BOUNDARY_RESOLUTION: usize = 100;
const METRICS_HISTORY_CAPACITY: usize = 512;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExperimentSpec {
    pub layer_sizes: Vec<usize>,
    pub activations: Vec<Activation>,
    pub features: Vec<InputFeature>,
    pub dataset: DatasetSpec,
    pub loss: LossKind,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub shuffle: bool,
    pub seed: u64,
}

impl ExperimentSpec {
    pub fn browser_default() -> Self {
        Self {
            layer_sizes: vec![2, 4, 1],
            activations: vec![Activation::Sigmoid, Activation::Sigmoid],
            features: vec![InputFeature::X1, InputFeature::X2],
            dataset: DatasetSpec::logical(DatasetKind::Xor),
            loss: LossKind::BinaryCrossEntropy,
            learning_rate: 0.03,
            batch_size: 4,
            shuffle: false,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MetricPoint {
    pub epoch: u64,
    pub update_count: u64,
    pub dataset_loss: f64,
    pub accuracy: f64,
    pub gradient_norm: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingChunk {
    pub updates_completed: usize,
    pub samples_processed: usize,
    pub epochs_completed: u64,
    pub update_count: u64,
    pub revision: u64,
    pub latest_metrics: MetricPoint,
    pub loss: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExperimentSnapshot {
    pub spec: ExperimentSpec,
    pub layer_sizes: Vec<usize>,
    pub activations: Vec<Activation>,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
    pub raw_dataset: RawDataset,
    pub metrics_history: Vec<MetricPoint>,
    pub epoch: u64,
    pub update_count: u64,
    pub samples_processed: u64,
    pub revision: u64,
    pub limits: NetworkLimits,
}

pub struct Experiment {
    spec: ExperimentSpec,
    feature_selection: FeatureSelection,
    network: Network,
    raw_dataset: RawDataset,
    training_examples: Vec<Example>,
    sample_order: Vec<usize>,
    cursor: usize,
    epoch: u64,
    update_count: u64,
    samples_processed: u64,
    last_gradient_norm: f64,
    metrics_history: VecDeque<MetricPoint>,
    shuffle_rng: ChaCha8Rng,
    revision: u64,
    limits: NetworkLimits,
}

impl Experiment {
    pub fn new(spec: ExperimentSpec, limits: NetworkLimits) -> Result<Self, EngineError> {
        validate_learning_controls(spec.learning_rate, spec.batch_size)?;
        let feature_selection = FeatureSelection::new(spec.features.clone())?;
        if spec.layer_sizes.first().copied() != Some(feature_selection.features().len()) {
            return Err(EngineError::ShapeMismatch(format!(
                "network expects {} inputs but {} features are selected",
                spec.layer_sizes.first().copied().unwrap_or_default(),
                feature_selection.features().len()
            )));
        }
        if spec.layer_sizes.last().copied() != Some(1) {
            return Err(EngineError::UnsupportedConfiguration(
                "playground experiments require one binary output".into(),
            ));
        }
        let network_config =
            NetworkConfig::with_limits(spec.layer_sizes.clone(), spec.activations.clone(), limits)?;
        spec.loss.validate_for(network_config.activations())?;
        if spec.loss == LossKind::BinaryCrossEntropy
            && network_config.activations().last() != Some(&Activation::Sigmoid)
        {
            return Err(EngineError::UnsupportedConfiguration(
                "binary cross-entropy requires sigmoid output".into(),
            ));
        }
        let raw_dataset = RawDataset::generate(spec.dataset.clone())?;
        let training_examples = raw_dataset
            .examples
            .iter()
            .map(|example| {
                Ok(Example {
                    inputs: feature_selection.transform(&example.inputs)?,
                    target: example.target.clone(),
                })
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        let network = Network::new(network_config, spec.seed);
        let mut sample_order = (0..training_examples.len()).collect::<Vec<_>>();
        let mut shuffle_rng = ChaCha8Rng::seed_from_u64(spec.seed ^ 0x9e37_79b9_7f4a_7c15);
        if spec.shuffle {
            sample_order.shuffle(&mut shuffle_rng);
        }
        let mut experiment = Self {
            spec,
            feature_selection,
            network,
            raw_dataset,
            training_examples,
            sample_order,
            cursor: 0,
            epoch: 0,
            update_count: 0,
            samples_processed: 0,
            last_gradient_norm: 0.0,
            metrics_history: VecDeque::with_capacity(METRICS_HISTORY_CAPACITY),
            shuffle_rng,
            revision: 0,
            limits,
        };
        let metrics = experiment.evaluate()?;
        experiment.push_metric(metrics);
        Ok(experiment)
    }

    pub fn browser_default() -> Result<Self, EngineError> {
        Self::new(ExperimentSpec::browser_default(), NetworkLimits::browser())
    }

    pub fn spec(&self) -> &ExperimentSpec {
        &self.spec
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Replace the experiment only after the complete proposed state validates.
    /// A rejected change leaves this experiment, including its revision, intact.
    pub fn reconfigure(
        &mut self,
        spec: ExperimentSpec,
        limits: NetworkLimits,
    ) -> Result<(), EngineError> {
        let mut replacement = Self::new(spec, limits)?;
        replacement.revision = self.revision.saturating_add(1);
        *self = replacement;
        Ok(())
    }

    pub fn snapshot(&self) -> ExperimentSnapshot {
        ExperimentSnapshot {
            spec: self.spec.clone(),
            layer_sizes: self.network.config().layer_sizes().to_vec(),
            activations: self.network.config().activations().to_vec(),
            weights: self.network.weights().to_vec(),
            biases: self.network.biases().to_vec(),
            raw_dataset: self.raw_dataset.clone(),
            metrics_history: self.metrics_history.iter().cloned().collect(),
            epoch: self.epoch,
            update_count: self.update_count,
            samples_processed: self.samples_processed,
            revision: self.revision,
            limits: self.limits,
        }
    }

    pub fn train_updates(&mut self, updates: usize) -> Result<TrainingChunk, EngineError> {
        if !(1..=MAX_UPDATES_PER_CHUNK).contains(&updates) {
            return Err(EngineError::ResourceLimit(format!(
                "training chunks must contain 1 to {MAX_UPDATES_PER_CHUNK} updates"
            )));
        }
        let start_epoch = self.epoch;
        let mut updates_completed = 0;
        let mut samples_processed = 0;
        let mut batch_loss = 0.0;
        for _ in 0..updates {
            if self.cursor >= self.training_examples.len() {
                self.finish_epoch()?;
            }
            let batch_len = self
                .spec
                .batch_size
                .min(self.training_examples.len() - self.cursor);
            let start = self.cursor;
            let end = start + batch_len;
            let order = &self.sample_order;
            let examples = &self.training_examples;
            let batch = (start..end).map(|position| &examples[order[position]]);
            let result = self.network.train_batch(
                batch,
                batch_len,
                self.spec.loss,
                self.spec.learning_rate,
            )?;
            batch_loss = result.loss;
            self.last_gradient_norm = 0.0;
            self.cursor = end;
            self.update_count += 1;
            self.revision = self.revision.saturating_add(1);
            self.samples_processed += batch_len as u64;
            updates_completed += 1;
            samples_processed += batch_len;
            self.last_gradient_norm = result.gradient_norm;
            if self.cursor == self.training_examples.len() {
                self.finish_epoch()?;
            }
        }
        let latest_metrics = self.metrics_history.back().cloned().unwrap_or(MetricPoint {
            epoch: self.epoch,
            update_count: self.update_count,
            dataset_loss: batch_loss,
            accuracy: 0.0,
            gradient_norm: self.last_gradient_norm,
        });
        Ok(TrainingChunk {
            updates_completed,
            samples_processed,
            epochs_completed: self.epoch - start_epoch,
            update_count: self.update_count,
            revision: self.revision,
            latest_metrics,
            loss: batch_loss,
        })
    }

    pub fn evaluate(&self) -> Result<MetricPoint, EngineError> {
        let mut total_loss = 0.0;
        let mut correct = 0usize;
        for example in &self.training_examples {
            let trace = self.network.forward(&example.inputs)?;
            total_loss += self.spec.loss.value(&trace.outputs, &example.target)?;
            let predicted = trace.outputs[0] >= 0.5;
            let expected = example.target[0] >= 0.5;
            correct += usize::from(predicted == expected);
        }
        Ok(MetricPoint {
            epoch: self.epoch,
            update_count: self.update_count,
            dataset_loss: total_loss / self.training_examples.len() as f64,
            accuracy: correct as f64 / self.training_examples.len() as f64,
            gradient_norm: self.last_gradient_norm,
        })
    }

    pub fn forward_batch(&self, points: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, EngineError> {
        if points.len() > 10_000 {
            return Err(EngineError::ResourceLimit(
                "batch inference is capped at 10,000 points".into(),
            ));
        }
        points
            .iter()
            .map(|point| {
                let transformed = self.feature_selection.transform(point)?;
                Ok(self.network.forward(&transformed)?.outputs)
            })
            .collect()
    }

    pub fn decision_boundary(&self, resolution: usize) -> Result<Vec<BoundaryPoint>, EngineError> {
        if !(2..=MAX_BOUNDARY_RESOLUTION).contains(&resolution) {
            return Err(EngineError::ResourceLimit(format!(
                "boundary resolution must be 2 to {MAX_BOUNDARY_RESOLUTION}"
            )));
        }
        let [x_min, x_max, y_min, y_max] = self.raw_dataset.bounds;
        let mut points = Vec::with_capacity(resolution * resolution);
        for row in 0..resolution {
            let y = y_min + (y_max - y_min) * row as f64 / (resolution - 1) as f64;
            for col in 0..resolution {
                let x = x_min + (x_max - x_min) * col as f64 / (resolution - 1) as f64;
                let features = self.feature_selection.transform(&[x, y])?;
                let probability = self.network.forward(&features)?.outputs[0];
                points.push(BoundaryPoint { x, y, probability });
            }
        }
        Ok(points)
    }

    pub fn reset(&mut self, seed: u64) -> Result<(), EngineError> {
        let config = NetworkConfig::with_limits(
            self.spec.layer_sizes.clone(),
            self.spec.activations.clone(),
            self.limits,
        )?;
        self.spec.seed = seed;
        self.network = Network::new(config, seed);
        self.cursor = 0;
        self.epoch = 0;
        self.update_count = 0;
        self.samples_processed = 0;
        self.last_gradient_norm = 0.0;
        self.metrics_history.clear();
        self.shuffle_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
        self.sample_order = (0..self.training_examples.len()).collect();
        if self.spec.shuffle {
            self.sample_order.shuffle(&mut self.shuffle_rng);
        }
        self.revision += 1;
        let metric = self.evaluate()?;
        self.push_metric(metric);
        Ok(())
    }

    fn finish_epoch(&mut self) -> Result<(), EngineError> {
        self.epoch += 1;
        self.cursor = 0;
        let metric = self.evaluate()?;
        self.push_metric(metric);
        if self.spec.shuffle {
            self.sample_order.shuffle(&mut self.shuffle_rng);
        }
        Ok(())
    }

    fn push_metric(&mut self, metric: MetricPoint) {
        if self.metrics_history.len() == METRICS_HISTORY_CAPACITY {
            self.metrics_history.pop_front();
        }
        self.metrics_history.push_back(metric);
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BoundaryPoint {
    pub x: f64,
    pub y: f64,
    pub probability: f64,
}

fn validate_learning_controls(learning_rate: f64, batch_size: usize) -> Result<(), EngineError> {
    if !(learning_rate.is_finite() && learning_rate > 0.0 && learning_rate < 100.0) {
        return Err(EngineError::InvalidValue(
            "learning rate must be in (0, 100)".into(),
        ));
    }
    if !(1..=MAX_BATCH_SIZE).contains(&batch_size) {
        return Err(EngineError::ResourceLimit(format!(
            "batch size must be 1 to {MAX_BATCH_SIZE}"
        )));
    }
    Ok(())
}

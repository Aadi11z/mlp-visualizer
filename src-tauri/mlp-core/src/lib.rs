mod activation;
mod dataset;
mod error;
mod experiment;
mod features;
mod network;

pub use activation::Activation;
pub use dataset::{DatasetKind, DatasetSpec, Example, RawDataset};
pub use error::{EngineError, NetworkConfigError};
pub use experiment::{Experiment, ExperimentSnapshot, ExperimentSpec, MetricPoint, TrainingChunk};
pub use features::{FeatureSelection, InputFeature};
pub use network::{ForwardTrace, Gradients, LossKind, Network, NetworkConfig, NetworkLimits};

#[cfg(target_arch = "wasm32")]
mod wasm;

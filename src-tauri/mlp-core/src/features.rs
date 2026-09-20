use crate::EngineError;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum InputFeature {
    X1,
    X2,
    X1Squared,
    X2Squared,
    X1X2,
    SinX1,
    SinX2,
}

impl InputFeature {
    pub const ALL: [Self; 7] = [
        Self::X1,
        Self::X2,
        Self::X1Squared,
        Self::X2Squared,
        Self::X1X2,
        Self::SinX1,
        Self::SinX2,
    ];

    pub fn evaluate(self, x1: f64, x2: f64) -> f64 {
        match self {
            Self::X1 => x1,
            Self::X2 => x2,
            Self::X1Squared => x1 * x1,
            Self::X2Squared => x2 * x2,
            Self::X1X2 => x1 * x2,
            Self::SinX1 => x1.sin(),
            Self::SinX2 => x2.sin(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureSelection {
    features: Vec<InputFeature>,
}

impl FeatureSelection {
    pub fn new(features: Vec<InputFeature>) -> Result<Self, EngineError> {
        if features.is_empty() {
            return Err(EngineError::InvalidValue(
                "select at least one input feature".into(),
            ));
        }
        if features.len() > InputFeature::ALL.len() {
            return Err(EngineError::ResourceLimit(
                "at most seven input features are supported".into(),
            ));
        }
        let unique = features.iter().copied().collect::<HashSet<_>>();
        if unique.len() != features.len() {
            return Err(EngineError::InvalidValue(
                "input features cannot be selected more than once".into(),
            ));
        }
        Ok(Self { features })
    }

    pub fn all() -> Self {
        Self {
            features: InputFeature::ALL.to_vec(),
        }
    }

    pub fn coordinates() -> Self {
        Self {
            features: vec![InputFeature::X1, InputFeature::X2],
        }
    }

    pub fn features(&self) -> &[InputFeature] {
        &self.features
    }

    pub fn transform(&self, point: &[f64]) -> Result<Vec<f64>, EngineError> {
        if point.len() != 2 || point.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::ShapeMismatch(
                "raw dataset points must contain two finite coordinates".into(),
            ));
        }
        let transformed = self
            .features
            .iter()
            .map(|feature| feature.evaluate(point[0], point[1]))
            .collect::<Vec<_>>();
        if transformed.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::NumericalFailure(
                "feature transformation produced a non-finite value".into(),
            ));
        }
        Ok(transformed)
    }
}

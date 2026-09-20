use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum Activation {
    #[serde(rename = "identity")]
    Identity,
    #[serde(rename = "sigmoid")]
    Sigmoid,
    #[serde(rename = "relu")]
    ReLU,
    #[serde(rename = "tanh")]
    Tanh,
    #[serde(rename = "step")]
    Step,
}

impl Activation {
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::Identity => x,
            Self::Sigmoid if x >= 0.0 => 1.0 / (1.0 + (-x).exp()),
            Self::Sigmoid => {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            }
            Self::ReLU => x.max(0.0),
            Self::Tanh => x.tanh(),
            Self::Step => f64::from(x >= 0.0),
        }
    }

    /// Derivatives are with respect to the pre-activation. ReLU chooses zero at x=0.
    pub fn derivative(self, x: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::Sigmoid => {
                let value = self.apply(x);
                value * (1.0 - value)
            }
            Self::ReLU => f64::from(x > 0.0),
            Self::Tanh => 1.0 - x.tanh().powi(2),
            Self::Step => 0.0,
        }
    }

    pub fn is_differentiable(self) -> bool {
        !matches!(self, Self::Step)
    }
}

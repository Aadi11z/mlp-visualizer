// src/mlp/activations.rs: Defines the Activation categories (functions and their derivatives) available in an enum

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum Activation {
    Identity,
    Sigmoid,
    ReLU,
    Tanh,
    Step,
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Identity => x,
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::ReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Activation::Tanh => x.tanh(),
            Activation::Step => {
                if x >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Identity => 1.0,
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Step => 0.0,
        }
    }
}

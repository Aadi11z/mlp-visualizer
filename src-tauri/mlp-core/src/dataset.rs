use crate::EngineError;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

const MAX_GENERATED_SAMPLES: usize = 1_000;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DatasetKind {
    Xor,
    And,
    Or,
    Moons,
    Spirals,
    Circles,
    Linear,
    Blobs,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct DatasetSpec {
    pub kind: DatasetKind,
    pub sample_count: usize,
    pub noise: f64,
    pub seed: u64,
}

impl DatasetSpec {
    pub fn logical(kind: DatasetKind) -> Self {
        Self {
            kind,
            sample_count: 4,
            noise: 0.0,
            seed: 0,
        }
    }

    pub fn generated(kind: DatasetKind, sample_count: usize, noise: f64, seed: u64) -> Self {
        Self {
            kind,
            sample_count,
            noise,
            seed,
        }
    }

    pub fn validate(&self) -> Result<(), EngineError> {
        match self.kind {
            DatasetKind::Xor | DatasetKind::And | DatasetKind::Or => {
                if self.sample_count != 4 || self.noise != 0.0 {
                    return Err(EngineError::InvalidValue(
                        "logical datasets require four points and zero noise".into(),
                    ));
                }
            }
            _ => {
                if !(4..=MAX_GENERATED_SAMPLES).contains(&self.sample_count)
                    || self.sample_count % 2 != 0
                {
                    return Err(EngineError::ResourceLimit(format!(
                        "generated datasets require an even sample count from 4 to {MAX_GENERATED_SAMPLES}"
                    )));
                }
                if !(self.noise.is_finite() && (0.0..=1.0).contains(&self.noise)) {
                    return Err(EngineError::InvalidValue(
                        "dataset noise must be finite and between 0 and 1".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Example {
    pub inputs: Vec<f64>,
    pub target: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RawDataset {
    pub spec: DatasetSpec,
    pub examples: Vec<Example>,
    pub bounds: [f64; 4],
}

impl RawDataset {
    pub fn generate(spec: DatasetSpec) -> Result<Self, EngineError> {
        spec.validate()?;
        let mut rng = ChaCha8Rng::seed_from_u64(spec.seed);
        let examples = match spec.kind {
            DatasetKind::Xor | DatasetKind::And | DatasetKind::Or => logical_examples(spec.kind),
            _ => generated_examples(&spec, &mut rng),
        };
        let bounds = match spec.kind {
            DatasetKind::Xor | DatasetKind::And | DatasetKind::Or => [0.0, 1.0, 0.0, 1.0],
            DatasetKind::Moons => [-1.5, 2.5, -1.5, 1.5],
            DatasetKind::Spirals | DatasetKind::Circles => [-1.5, 1.5, -1.5, 1.5],
            DatasetKind::Linear | DatasetKind::Blobs => [-1.5, 1.5, -1.5, 1.5],
        };
        Ok(Self {
            spec,
            examples,
            bounds,
        })
    }
}

fn logical_examples(kind: DatasetKind) -> Vec<Example> {
    let points = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    points
        .into_iter()
        .map(|[x1, x2]| {
            let label = match kind {
                DatasetKind::Xor => (x1 as u8) ^ (x2 as u8),
                DatasetKind::And => (x1 as u8) & (x2 as u8),
                DatasetKind::Or => (x1 as u8) | (x2 as u8),
                _ => unreachable!(),
            };
            Example {
                inputs: vec![x1, x2],
                target: vec![f64::from(label)],
            }
        })
        .collect()
}

fn generated_examples(spec: &DatasetSpec, rng: &mut ChaCha8Rng) -> Vec<Example> {
    let class_count = spec.sample_count / 2;
    let mut examples = Vec::with_capacity(spec.sample_count);
    for class in 0..2 {
        for index in 0..class_count {
            let t = (index as f64 + 0.5) / class_count as f64;
            let (mut x1, mut x2) = match spec.kind {
                DatasetKind::Moons => {
                    let angle = std::f64::consts::PI * t;
                    if class == 0 {
                        (angle.cos(), angle.sin())
                    } else {
                        (1.0 - angle.cos(), 0.5 - angle.sin())
                    }
                }
                DatasetKind::Spirals => {
                    let radius = 0.1 + 0.8 * t;
                    let angle = 3.5 * std::f64::consts::PI * t
                        + if class == 0 {
                            0.0
                        } else {
                            std::f64::consts::PI
                        };
                    (radius * angle.cos(), radius * angle.sin())
                }
                DatasetKind::Circles => {
                    let angle = 2.0 * std::f64::consts::PI * t;
                    let radius = if class == 0 { 0.35 } else { 0.8 };
                    (radius * angle.cos(), radius * angle.sin())
                }
                DatasetKind::Linear => {
                    let x = rng.random_range(-1.0_f64..1.0_f64);
                    let split: f64 = -x;
                    let y = if class == 0 {
                        rng.random_range(-1.0..split.clamp(-1.0, 1.0))
                    } else {
                        rng.random_range(split.clamp(-1.0, 1.0)..1.0)
                    };
                    (x, y)
                }
                DatasetKind::Blobs => {
                    let mut nx: f64 = StandardNormal.sample(rng);
                    let mut ny: f64 = StandardNormal.sample(rng);
                    nx *= 0.18;
                    ny *= 0.18;
                    (if class == 0 { -0.55 + nx } else { 0.55 + nx }, ny)
                }
                _ => unreachable!(),
            };
            if spec.kind != DatasetKind::Blobs || spec.noise > 0.0 {
                let nx: f64 = StandardNormal.sample(rng);
                let ny: f64 = StandardNormal.sample(rng);
                x1 += nx * spec.noise;
                x2 += ny * spec.noise;
            }
            examples.push(Example {
                inputs: vec![x1, x2],
                target: vec![class as f64],
            });
        }
    }
    examples
}

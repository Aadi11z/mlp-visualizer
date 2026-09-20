use axum::http::StatusCode;
use mlp_core::{
    Activation, DatasetKind, DatasetSpec, EngineError, LossKind, Network, NetworkConfig, RawDataset,
};
use serde::Serialize;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;

const DEFAULT_LAYERS: &[usize] = &[2, 4, 1];
const DEFAULT_LR: f64 = 0.5;
pub const DEFAULT_SESSION_ID: &str = "default";
const DEFAULT_SESSION_TTL_SECS: u64 = 60 * 60;
const DEFAULT_MAX_SESSIONS: usize = 1_000;
type TrainingSample = (Vec<f64>, Vec<f64>);

fn xor_dataset() -> Vec<(Vec<f64>, Vec<f64>)> {
    raw_dataset(DatasetKind::Xor).expect("built-in XOR dataset must be valid")
}

fn raw_dataset(kind: DatasetKind) -> Result<Vec<TrainingSample>, EngineError> {
    let spec = match kind {
        DatasetKind::Xor | DatasetKind::And | DatasetKind::Or => DatasetSpec::logical(kind),
        _ => DatasetSpec::generated(kind, 20, 0.0, 42),
    };
    Ok(RawDataset::generate(spec)?
        .examples
        .into_iter()
        .map(|example| (example.inputs, example.target))
        .collect())
}

fn default_network(seed: u64) -> Network {
    let config = NetworkConfig::new(DEFAULT_LAYERS.to_vec(), vec![Activation::Sigmoid; 2])
        .expect("the built-in network config must be valid");
    Network::new(config, seed)
}

fn map_engine_error(error: EngineError) -> (StatusCode, String) {
    let status = match error {
        EngineError::ResourceLimit(_) => StatusCode::PAYLOAD_TOO_LARGE,
        _ => StatusCode::BAD_REQUEST,
    };
    (status, error.to_string())
}

pub struct MlpSession {
    net: Network,
    lr: f64,
    step_count: u64,
    last_loss: Option<f64>,
    dataset: Vec<TrainingSample>,
    dataset_name: String,
    initialization_seed: u64,
    last_accessed: Instant,
}

impl MlpSession {
    pub fn new() -> Self {
        let initialization_seed = rand::random::<u64>();
        Self {
            net: default_network(initialization_seed),
            lr: DEFAULT_LR,
            step_count: 0,
            last_loss: None,
            dataset: xor_dataset(),
            dataset_name: "XOR".to_string(),
            initialization_seed,
            last_accessed: Instant::now(),
        }
    }

    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }

    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.last_accessed.elapsed() > ttl
    }

    pub fn response(&self, session_id: &str) -> StateResponse {
        StateResponse {
            session_id: session_id.to_string(),
            layer_sizes: self.net.config().layer_sizes().to_vec(),
            activations: self.net.config().activations().to_vec(),
            weights: self.net.weights().to_vec(),
            biases: self.net.biases().to_vec(),
            learning_rate: self.lr,
            step_count: self.step_count,
            last_loss: self.last_loss,
            dataset_name: self.dataset_name.clone(),
            dataset: self.dataset.clone(),
        }
    }

    pub fn train_steps(&mut self, count: u32) -> Result<StepResponse, (StatusCode, String)> {
        if count == 0 {
            return Err((StatusCode::BAD_REQUEST, "count must be >= 1".into()));
        }
        if count > 10_000 {
            return Err((StatusCode::BAD_REQUEST, "count must be <= 10000".into()));
        }

        self.touch();
        let mut losses = Vec::with_capacity(count as usize);
        let start_step = self.step_count;
        let lr = self.lr;

        for i in 0..count {
            let idx = ((start_step + i as u64) as usize) % self.dataset.len();
            let (inputs, target) = {
                let (inputs, target) = &self.dataset[idx];
                (inputs.clone(), target.clone())
            };
            let res = self
                .net
                .train_sample(&inputs, &target, LossKind::MeanSquaredError, lr)
                .map_err(map_engine_error)?;
            losses.push(res.loss);
        }

        let last_loss = *losses.last().unwrap();
        let average_loss = losses.iter().sum::<f64>() / losses.len() as f64;
        self.step_count += count as u64;
        self.last_loss = Some(last_loss);

        Ok(StepResponse {
            steps_taken: count,
            step_count: self.step_count,
            last_loss,
            average_loss,
            weights: self.net.weights().to_vec(),
            biases: self.net.biases().to_vec(),
        })
    }

    pub fn randomize(&mut self, seed: Option<u64>) {
        self.initialization_seed = seed.unwrap_or_else(rand::random);
        self.net = default_network(self.initialization_seed);
        self.step_count = 0;
        self.last_loss = None;
        self.touch();
    }

    pub fn set_learning_rate(&mut self, lr: f64) -> Result<(), (StatusCode, String)> {
        if !(lr.is_finite() && lr > 0.0 && lr < 100.0) {
            return Err((StatusCode::BAD_REQUEST, "lr must be in (0, 100)".into()));
        }
        self.lr = lr;
        self.touch();
        Ok(())
    }

    pub fn set_dataset(&mut self, name: &str) -> Result<(), (StatusCode, String)> {
        let kind = match name {
            "XOR" => DatasetKind::Xor,
            "half-moons" => DatasetKind::Moons,
            "spirals" => DatasetKind::Spirals,
            _ => return Err((StatusCode::BAD_REQUEST, format!("unknown dataset: {name}"))),
        };
        let data = raw_dataset(kind).map_err(map_engine_error)?;
        self.dataset = data;
        self.dataset_name = name.to_string();
        self.initialization_seed = rand::random();
        self.net = default_network(self.initialization_seed);
        self.step_count = 0;
        self.last_loss = None;
        self.touch();
        Ok(())
    }

    pub fn forward(&self, inputs: &[f64]) -> Result<ForwardResponse, (StatusCode, String)> {
        let expected = self.net.config().layer_sizes()[0];
        if inputs.len() != expected {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("inputs must have length {}", expected),
            ));
        }
        let res = self.net.forward(inputs).map_err(map_engine_error)?;
        Ok(ForwardResponse {
            outputs: res.outputs,
            layer_outputs: res.activations,
        })
    }
}

impl Default for MlpSession {
    fn default() -> Self {
        Self::new()
    }
}

pub type SharedSession = Arc<Mutex<MlpSession>>;
pub type SharedState = Arc<SessionStore>;

pub struct SessionStore {
    sessions: Mutex<HashMap<String, SharedSession>>,
    session_ttl: Duration,
    max_sessions: usize,
}

impl SessionStore {
    pub fn new(session_ttl: Duration, max_sessions: usize) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            session_ttl,
            max_sessions,
        }
    }

    pub fn from_env() -> Self {
        let session_ttl = Duration::from_secs(
            std::env::var("MLP_SESSION_TTL_SECS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_SESSION_TTL_SECS),
        );
        let max_sessions = std::env::var("MLP_MAX_SESSIONS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_MAX_SESSIONS);
        Self::new(session_ttl, max_sessions)
    }

    pub async fn get_or_create(
        &self,
        session_id: &str,
    ) -> Result<SharedSession, (StatusCode, String)> {
        validate_session_id(session_id)?;
        let mut sessions = self.sessions.lock().await;
        if let Some(session) = sessions.get(session_id) {
            return Ok(session.clone());
        }
        if sessions.len() >= self.max_sessions {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "session capacity reached".to_string(),
            ));
        }
        let session = Arc::new(Mutex::new(MlpSession::new()));
        sessions.insert(session_id.to_string(), session.clone());
        Ok(session)
    }

    pub async fn prune_expired(&self) {
        let entries = {
            let sessions = self.sessions.lock().await;
            sessions
                .iter()
                .map(|(id, session)| (id.clone(), session.clone()))
                .collect::<Vec<_>>()
        };
        let mut expired = Vec::new();
        for (id, session) in entries {
            if session.lock().await.is_expired(self.session_ttl) {
                expired.push(id);
            }
        }
        if expired.is_empty() {
            return;
        }
        let mut sessions = self.sessions.lock().await;
        for id in expired {
            sessions.remove(&id);
        }
    }
}

pub fn validate_session_id(session_id: &str) -> Result<(), (StatusCode, String)> {
    let valid = !session_id.is_empty()
        && session_id.len() <= 64
        && session_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_');
    if valid {
        Ok(())
    } else {
        Err((
            StatusCode::BAD_REQUEST,
            "session_id must be 1-64 ASCII letters, digits, hyphens, or underscores".into(),
        ))
    }
}

#[derive(Serialize)]
pub struct StateResponse {
    pub session_id: String,
    pub layer_sizes: Vec<usize>,
    pub activations: Vec<Activation>,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
    pub learning_rate: f64,
    pub step_count: u64,
    pub last_loss: Option<f64>,
    pub dataset_name: String,
    pub dataset: Vec<(Vec<f64>, Vec<f64>)>,
}

#[derive(Serialize)]
pub struct StepResponse {
    pub steps_taken: u32,
    pub step_count: u64,
    pub last_loss: f64,
    pub average_loss: f64,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
}

#[derive(Serialize)]
pub struct ForwardResponse {
    pub outputs: Vec<f64>,
    pub layer_outputs: Vec<Vec<f64>>,
}

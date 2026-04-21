use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

mod mlp;
use mlp::activations::Activation;
use mlp::network::{Network, NetworkConfig};

const DEFAULT_LAYERS: &[usize] = &[2, 4, 1];
const DEFAULT_LR: f64 = 0.5;

fn xor_dataset() -> Vec<(Vec<f64>, Vec<f64>)> {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}

fn default_network() -> Network {
    let cfg = NetworkConfig {
        layer_sizes: DEFAULT_LAYERS.to_vec(),
        activations: vec![Activation::Sigmoid, Activation::Sigmoid],
    };
    Network::from_config(cfg)
}

fn seeded_network(seed: u64) -> Network {
    // Rebuild a network with deterministic weights from the given seed.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let layer_sizes = DEFAULT_LAYERS.to_vec();
    let activations = vec![Activation::Sigmoid, Activation::Sigmoid];
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    for i in 0..(layer_sizes.len() - 1) {
        let in_size = layer_sizes[i];
        let out_size = layer_sizes[i + 1];
        let scale = (2.0 / in_size as f64).sqrt();
        let mut layer_w = vec![vec![0.0; in_size]; out_size];
        let layer_b = vec![0.0; out_size];
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
        config: NetworkConfig { layer_sizes, activations },
        weights,
        biases,
    }
}

struct AppState {
    net: Mutex<Network>,
    lr: Mutex<f64>,
    step_count: Mutex<u64>,
    last_loss: Mutex<Option<f64>>,
    dataset: Vec<(Vec<f64>, Vec<f64>)>,
}

impl AppState {
    fn new() -> Self {
        Self {
            net: Mutex::new(default_network()),
            lr: Mutex::new(DEFAULT_LR),
            step_count: Mutex::new(0),
            last_loss: Mutex::new(None),
            dataset: xor_dataset(),
        }
    }
}

type SharedState = Arc<AppState>;

#[derive(Serialize)]
struct StateResponse {
    layer_sizes: Vec<usize>,
    activations: Vec<Activation>,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    learning_rate: f64,
    step_count: u64,
    last_loss: Option<f64>,
    dataset_name: String,
    dataset: Vec<(Vec<f64>, Vec<f64>)>,
}

async fn get_state(State(state): State<SharedState>) -> Json<StateResponse> {
    let net = state.net.lock().await;
    let lr = *state.lr.lock().await;
    let step_count = *state.step_count.lock().await;
    let last_loss = *state.last_loss.lock().await;
    Json(StateResponse {
        layer_sizes: net.config.layer_sizes.clone(),
        activations: net.config.activations.clone(),
        weights: net.weights.clone(),
        biases: net.biases.clone(),
        learning_rate: lr,
        step_count,
        last_loss,
        dataset_name: "XOR".to_string(),
        dataset: state.dataset.clone(),
    })
}

#[derive(Deserialize)]
struct StepRequest {
    #[serde(default = "default_step_count")]
    count: u32,
}
fn default_step_count() -> u32 { 1 }

#[derive(Serialize)]
struct StepResponse {
    steps_taken: u32,
    step_count: u64,
    last_loss: f64,
    average_loss: f64,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
}

async fn post_step(
    State(state): State<SharedState>,
    Json(req): Json<StepRequest>,
) -> Result<Json<StepResponse>, (StatusCode, String)> {
    if req.count == 0 {
        return Err((StatusCode::BAD_REQUEST, "count must be >= 1".into()));
    }
    if req.count > 10_000 {
        return Err((StatusCode::BAD_REQUEST, "count must be <= 10000".into()));
    }

    let mut net = state.net.lock().await;
    let lr = *state.lr.lock().await;
    let mut losses = Vec::with_capacity(req.count as usize);
    let start_step = *state.step_count.lock().await;

    for i in 0..req.count {
        let idx = ((start_step + i as u64) as usize) % state.dataset.len();
        let (inputs, target) = &state.dataset[idx];
        let res = net.train_step(inputs, target, lr);
        losses.push(res.loss);
    }

    let last_loss = *losses.last().unwrap();
    let avg = losses.iter().sum::<f64>() / losses.len() as f64;

    *state.step_count.lock().await += req.count as u64;
    *state.last_loss.lock().await = Some(last_loss);

    Ok(Json(StepResponse {
        steps_taken: req.count,
        step_count: *state.step_count.lock().await,
        last_loss,
        average_loss: avg,
        weights: net.weights.clone(),
        biases: net.biases.clone(),
    }))
}

#[derive(Deserialize)]
struct RandomizeRequest {
    seed: Option<u64>,
}

async fn post_randomize(
    State(state): State<SharedState>,
    Json(req): Json<RandomizeRequest>,
) -> Json<StateResponse> {
    let new_net = match req.seed {
        Some(s) => seeded_network(s),
        None => default_network(),
    };
    *state.net.lock().await = new_net;
    *state.step_count.lock().await = 0;
    *state.last_loss.lock().await = None;
    get_state(State(state)).await
}

#[derive(Deserialize)]
struct SetLrRequest {
    lr: f64,
}

async fn post_set_lr(
    State(state): State<SharedState>,
    Json(req): Json<SetLrRequest>,
) -> Result<Json<StateResponse>, (StatusCode, String)> {
    if !(req.lr.is_finite() && req.lr > 0.0 && req.lr < 100.0) {
        return Err((StatusCode::BAD_REQUEST, "lr must be in (0, 100)".into()));
    }
    *state.lr.lock().await = req.lr;
    Ok(get_state(State(state)).await)
}

#[derive(Deserialize)]
struct ForwardRequest {
    inputs: Vec<f64>,
}

#[derive(Serialize)]
struct ForwardResponse {
    outputs: Vec<f64>,
    layer_outputs: Vec<Vec<f64>>,
}

async fn post_forward(
    State(state): State<SharedState>,
    Json(req): Json<ForwardRequest>,
) -> Result<Json<ForwardResponse>, (StatusCode, String)> {
    let net = state.net.lock().await;
    let expected = net.config.layer_sizes[0];
    if req.inputs.len() != expected {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("inputs must have length {}", expected),
        ));
    }
    let res = net.forward(&req.inputs);
    Ok(Json(ForwardResponse {
        outputs: res.outputs,
        layer_outputs: res.layer_outputs,
    }))
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status":"ok"})))
}

#[tokio::main]
async fn main() {
    let state: SharedState = Arc::new(AppState::new());

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/state", get(get_state))
        .route("/step", post(post_step))
        .route("/randomize", post(post_randomize))
        .route("/set_lr", post(post_set_lr))
        .route("/forward", post(post_forward))
        .layer(cors)
        .with_state(state);

    let port: u16 = std::env::var("MLP_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9000);
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    println!("mlp-server listening on http://{addr}");
    axum::serve(listener, app).await.expect("serve");
}

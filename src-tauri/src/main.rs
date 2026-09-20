use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use mlp_server::session::{SessionStore, SharedState, StateResponse, DEFAULT_SESSION_ID};
use serde::Deserialize;
use std::{sync::Arc, time::Duration};
use tokio::time;

#[derive(Deserialize, Default)]
struct SessionQuery {
    session_id: Option<String>,
}

impl SessionQuery {
    fn id(&self) -> &str {
        self.session_id.as_deref().unwrap_or(DEFAULT_SESSION_ID)
    }
}

async fn get_state(
    State(state): State<SharedState>,
    Query(query): Query<SessionQuery>,
) -> Result<Json<StateResponse>, (StatusCode, String)> {
    let session = state.get_or_create(query.id()).await?;
    let mut session = session.lock().await;
    session.touch();
    Ok(Json(session.response(query.id())))
}

#[derive(Deserialize)]
struct StepRequest {
    #[serde(default = "default_step_count")]
    count: u32,
}
fn default_step_count() -> u32 {
    1
}

async fn post_step(
    State(state): State<SharedState>,
    Query(query): Query<SessionQuery>,
    Json(req): Json<StepRequest>,
) -> Result<Json<mlp_server::session::StepResponse>, (StatusCode, String)> {
    let session = state.get_or_create(query.id()).await?;
    let mut session = session.lock().await;
    Ok(Json(session.train_steps(req.count)?))
}

#[derive(Deserialize)]
struct RandomizeRequest {
    seed: Option<u64>,
}

async fn post_randomize(
    State(state): State<SharedState>,
    Query(query): Query<SessionQuery>,
    Json(req): Json<RandomizeRequest>,
) -> Result<Json<mlp_server::session::StateResponse>, (StatusCode, String)> {
    let session = state.get_or_create(query.id()).await?;
    let mut session = session.lock().await;
    session.randomize(req.seed);
    Ok(Json(session.response(query.id())))
}

#[derive(Deserialize)]
struct SetLrRequest {
    lr: f64,
}

async fn post_set_lr(
    State(state): State<SharedState>,
    Query(query): Query<SessionQuery>,
    Json(req): Json<SetLrRequest>,
) -> Result<Json<mlp_server::session::StateResponse>, (StatusCode, String)> {
    let session = state.get_or_create(query.id()).await?;
    let mut session = session.lock().await;
    session.set_learning_rate(req.lr)?;
    Ok(Json(session.response(query.id())))
}

#[derive(Deserialize)]
struct SetDatasetRequest {
    name: String,
}

async fn post_set_dataset(
    State(state): State<SharedState>,
    Query(query): Query<SessionQuery>,
    Json(req): Json<SetDatasetRequest>,
) -> Result<Json<mlp_server::session::StateResponse>, (StatusCode, String)> {
    let session = state.get_or_create(query.id()).await?;
    let mut session = session.lock().await;
    session.set_dataset(&req.name)?;
    Ok(Json(session.response(query.id())))
}

#[derive(Deserialize)]
struct ForwardRequest {
    inputs: Vec<f64>,
}

async fn post_forward(
    State(state): State<SharedState>,
    Query(query): Query<SessionQuery>,
    Json(req): Json<ForwardRequest>,
) -> Result<Json<mlp_server::session::ForwardResponse>, (StatusCode, String)> {
    let session = state.get_or_create(query.id()).await?;
    let mut session = session.lock().await;
    session.touch();
    Ok(Json(session.forward(&req.inputs)?))
}

async fn get_health() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status":"ok"})))
}

#[tokio::main]
async fn main() {
    let state: SharedState = Arc::new(SessionStore::from_env());
    let cleanup_state = state.clone();
    tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(300));
        loop {
            interval.tick().await;
            cleanup_state.prune_expired().await;
        }
    });

    let app = Router::new()
        .route("/get_health", get(get_health))
        .route("/state", get(get_state))
        .route("/step", post(post_step))
        .route("/randomize", post(post_randomize))
        .route("/set_lr", post(post_set_lr))
        .route("/set_dataset", post(post_set_dataset))
        .route("/forward", post(post_forward))
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

use crate::{Experiment, ExperimentSpec, NetworkLimits};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct BrowserEngine {
    experiment: Experiment,
    limits: NetworkLimits,
}

#[wasm_bindgen]
impl BrowserEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(spec_json: &str) -> Result<BrowserEngine, JsValue> {
        Self::with_limits(spec_json, NetworkLimits::browser())
    }

    #[wasm_bindgen(js_name = browserDefault)]
    pub fn browser_default() -> Result<BrowserEngine, JsValue> {
        Self::with_spec(ExperimentSpec::browser_default(), NetworkLimits::browser())
    }

    #[wasm_bindgen(js_name = measurementDefault)]
    pub fn measurement_default() -> Result<BrowserEngine, JsValue> {
        Self::with_spec(
            ExperimentSpec::browser_default(),
            NetworkLimits::measurement_candidates(),
        )
    }

    #[wasm_bindgen(js_name = measurement)]
    pub fn measurement(spec_json: &str) -> Result<BrowserEngine, JsValue> {
        Self::with_limits(spec_json, NetworkLimits::measurement_candidates())
    }

    pub fn configure(&mut self, spec_json: &str) -> Result<(), JsValue> {
        let spec: ExperimentSpec = serde_json::from_str(spec_json).map_err(js_error)?;
        self.experiment
            .reconfigure(spec, self.limits)
            .map_err(js_error)
    }

    pub fn reset(&mut self, seed: u64) -> Result<(), JsValue> {
        self.experiment.reset(seed).map_err(js_error)
    }

    pub fn revision(&self) -> String {
        self.experiment.revision().to_string()
    }

    pub fn train_chunk(&mut self, updates: u32) -> Result<String, JsValue> {
        let chunk = self
            .experiment
            .train_updates(updates as usize)
            .map_err(js_error)?;
        serde_json::to_string(&chunk).map_err(js_error)
    }

    pub fn snapshot(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.experiment.snapshot()).map_err(js_error)
    }

    pub fn evaluate(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.experiment.evaluate().map_err(js_error)?).map_err(js_error)
    }

    pub fn forward_batch(&self, points_json: &str) -> Result<String, JsValue> {
        let points: Vec<Vec<f64>> = serde_json::from_str(points_json).map_err(js_error)?;
        let outputs = self.experiment.forward_batch(&points).map_err(js_error)?;
        serde_json::to_string(&outputs).map_err(js_error)
    }

    pub fn decision_boundary(&self, resolution: u32) -> Result<String, JsValue> {
        let points = self
            .experiment
            .decision_boundary(resolution as usize)
            .map_err(js_error)?;
        serde_json::to_string(&points).map_err(js_error)
    }
}

impl BrowserEngine {
    fn with_limits(spec_json: &str, limits: NetworkLimits) -> Result<Self, JsValue> {
        let spec: ExperimentSpec = serde_json::from_str(spec_json).map_err(js_error)?;
        Self::with_spec(spec, limits)
    }

    fn with_spec(spec: ExperimentSpec, limits: NetworkLimits) -> Result<Self, JsValue> {
        let experiment = Experiment::new(spec, limits).map_err(js_error)?;
        Ok(Self { experiment, limits })
    }
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

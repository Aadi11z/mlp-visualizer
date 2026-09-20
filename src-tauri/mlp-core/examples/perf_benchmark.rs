use mlp_core::{
    Activation, DatasetKind, DatasetSpec, Example, Experiment, ExperimentSpec, FeatureSelection,
    InputFeature, LossKind, Network, NetworkConfig, NetworkLimits, RawDataset,
};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{hint::black_box, time::Instant};

const WARMUP_RUNS: usize = 1;
const MEASURED_RUNS: usize = 5;
const EPOCHS_PER_RUN: usize = 3;

fn main() {
    println!("native release benchmark; warmups={WARMUP_RUNS}; samples={MEASURED_RUNS}; epochs_per_run={EPOCHS_PER_RUN}");
    for width in [8, 16, 32] {
        let layers = std::iter::once(7)
            .chain(std::iter::repeat(width).take(6))
            .chain(std::iter::once(1))
            .collect::<Vec<_>>();
        let activations = vec![Activation::Tanh; layers.len() - 1];
        let config = NetworkConfig::with_limits(
            layers.clone(),
            activations.clone(),
            NetworkLimits::default(),
        )
        .expect("candidate architecture must fit core benchmark limits");

        for sample_count in [200, 1_000] {
            let dataset = training_examples(sample_count);
            let mut elapsed = Vec::with_capacity(MEASURED_RUNS);
            for run in 0..(WARMUP_RUNS + MEASURED_RUNS) {
                let mut network = Network::new(config.clone(), 42);
                let mut rng = ChaCha8Rng::seed_from_u64(7_331);
                let mut order = (0..dataset.len()).collect::<Vec<_>>();
                let start = Instant::now();
                for _ in 0..EPOCHS_PER_RUN {
                    order.shuffle(&mut rng);
                    for &sample_index in &order {
                        let example = &dataset[sample_index];
                        let result = network
                            .train_sample(
                                &example.inputs,
                                &example.target,
                                LossKind::MeanSquaredError,
                                0.03,
                            )
                            .expect("training example must be valid");
                        black_box(result.loss);
                    }
                }
                let duration = start.elapsed().as_secs_f64() * 1_000.0;
                if run >= WARMUP_RUNS {
                    elapsed.push(duration);
                }
            }
            let updates = sample_count * EPOCHS_PER_RUN;
            println!(
                "training_only width={width} layers=6 parameters={} samples={sample_count} updates={updates} median_ms={:.3} p95_ms={:.3} updates_per_sec={:.0}",
                config.parameter_count(),
                percentile(&elapsed, 0.50),
                percentile(&elapsed, 0.95),
                updates as f64 / (percentile(&elapsed, 0.50) / 1_000.0),
            );
        }
    }

    benchmark_inspection();
}

fn training_examples(sample_count: usize) -> Vec<Example> {
    let raw = RawDataset::generate(DatasetSpec::generated(
        DatasetKind::Moons,
        sample_count,
        0.1,
        42,
    ))
    .expect("benchmark dataset must be valid");
    let features = FeatureSelection::all();
    raw.examples
        .iter()
        .map(|example| Example {
            inputs: features.transform(&example.inputs).unwrap(),
            target: example.target.clone(),
        })
        .collect()
}

fn benchmark_inspection() {
    let layers = vec![7, 8, 8, 8, 8, 8, 8, 1];
    let spec = ExperimentSpec {
        layer_sizes: layers.clone(),
        activations: vec![Activation::Tanh; layers.len() - 1],
        features: InputFeature::ALL.to_vec(),
        dataset: DatasetSpec::generated(DatasetKind::Moons, 200, 0.1, 42),
        loss: LossKind::MeanSquaredError,
        learning_rate: 0.03,
        batch_size: 1,
        shuffle: false,
        seed: 42,
    };

    let started = Instant::now();
    let mut experiment = Experiment::new(spec.clone(), NetworkLimits::browser()).unwrap();
    println!(
        "experiment_startup_ms={:.3}",
        started.elapsed().as_secs_f64() * 1_000.0
    );

    let started = Instant::now();
    black_box(experiment.evaluate().unwrap());
    println!(
        "evaluate_ms={:.3}",
        started.elapsed().as_secs_f64() * 1_000.0
    );

    let started = Instant::now();
    black_box(experiment.snapshot());
    println!(
        "snapshot_ms={:.3}",
        started.elapsed().as_secs_f64() * 1_000.0
    );

    let started = Instant::now();
    black_box(experiment.train_updates(200).unwrap());
    println!(
        "train_plus_epoch_metrics_200_updates_ms={:.3}",
        started.elapsed().as_secs_f64() * 1_000.0
    );

    for resolution in [32, 50, 100] {
        let started = Instant::now();
        black_box(experiment.decision_boundary(resolution).unwrap());
        println!(
            "boundary_{resolution}x{resolution}_ms={:.3}",
            started.elapsed().as_secs_f64() * 1_000.0
        );
    }
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

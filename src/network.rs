//                                    b1 
// Input (0) -> w1 \                  |
//                  -> Hidden Neuron (h1) -> w5    
// Input (1) -> w3 /                            \
//                                               --> Output Neuron (y) Error = y^ - y
// Input (0) -> w2 \                            /                   |
//                  -> Hidden Neuron (h2) -> w6                     b3
// Input (1) -> w4 /                  |                             
//                                    b2                            
use rand::Rng;

fn sigmoid(x: f64) -> f64{
    1.0/(1.0 + (-x).exp())
}
pub struct InputLayer{
    pub inputs: Vec<f64>,
}

impl InputLayer{
    fn new() -> Self{ 
        InputLayer { inputs: Vec::new() }
    }
}

pub struct Layer{
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl Layer{
    pub fn new() -> Self{
        Layer { weights: Vec::new(), biases: Vec::new() }
    }

    pub fn forward_prop(&mut self, inputs: &mut InputLayer) -> Vec<f64> {
        let mut outputs = Vec::new();

        for (neuron_weights, bias) in self.weights.iter().zip(self.biases.iter()) {
            let h: f64 = neuron_weights.iter()
                .zip(inputs.inputs.iter())
                .map(|(w, i)| w * i)
                .sum::<f64>() + bias;
            outputs.push(sigmoid(h));
        }
        outputs
    }

    

    pub fn backward_prop(&mut self, inputs: &mut InputLayer) {

    }
}

// impl Neuron{
//     pub fn new(input_size: u64) -> Self{
//         let mut rng = rand::rng();
//         let weights: Vec<f64> = (0..input_size)
//             .map(|_| rng.random_range(-1.0..1.0))
//             .collect();

//         let bias = rng.random_range(-1.0..1.0);

//         Self { weights, bias, output: 0.0 }
//     }
//     pub fn forward(&mut self, inputs: &[f64]) -> f64 {
//         let h: f64 = self.weights.iter()
//             .zip(inputs)
//             .map(|(w, i)| w * i)
//             .sum::<f64>() + self.bias;

            
//         self.output = sigmoid(h);
//         self.output
//     }
//     pub fn backprop(&mut self) -> {

//     }
// }
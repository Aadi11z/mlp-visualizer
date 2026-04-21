// Types mirror the Rust server's JSON responses.

export type Activation = 'Identity' | 'Sigmoid' | 'ReLU' | 'Tanh' | 'Step';

export type NetState = {
  layer_sizes: number[];
  activations: Activation[];
  weights: number[][][]; // [layer][out_neuron][in_neuron]
  biases: number[][];    // [layer][neuron]
  learning_rate: number;
  step_count: number;
  last_loss: number | null;
  dataset_name: string;
  dataset: [number[], number[]][];
};

export type StepResult = {
  steps_taken: number;
  step_count: number;
  last_loss: number;
  average_loss: number;
  weights: number[][][];
  biases: number[][];
};

export type ForwardResult = {
  outputs: number[];
  layer_outputs: number[][];
};

export type AgentTrace = {
  tool: string;
  input: Record<string, unknown>;
  result: Record<string, unknown>;
};

export type AgentAnswer = {
  answer: string;
  trace: AgentTrace[];
  stop_reason: string | null;
};

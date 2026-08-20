use crate::{Neuron, Value};
use rand::RngExt;

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(
        n_inputs: usize,
        n_outputs: usize,
        activation: bool,
        rng: &mut impl RngExt,
    ) -> Layer {
        Layer {
            neurons: (0..n_outputs)
                .map(|_| Neuron::new(n_inputs, activation, rng))
                .collect(),
        }
    }

    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(inputs)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

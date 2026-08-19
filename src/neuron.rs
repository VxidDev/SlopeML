use crate::RngExt;
use crate::Value;

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(n_inputs: usize, rng: &mut impl RngExt) -> Neuron {
        Neuron {
            weights: (0..n_inputs)
                .map(|_| Value::new(rng.random_range(-1.0..1.0)))
                .collect(),
            bias: Value::new(rng.random_range(-1.0..1.0)),
        }
    }

    pub fn forward(&self, inputs: &[Value]) -> Value {
        let mut sum = self.bias.clone();

        for (w, x) in self.weights.iter().zip(inputs) {
            sum = sum + w * x;
        }

        sum
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.weights.clone();
        p.push(self.bias.clone());
        p
    }
}

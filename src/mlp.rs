use crate::{Layer, Value};
use rand::RngExt;

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(n_inputs: usize, layer_sizes: &[usize], rng: &mut impl RngExt) -> MLP {
        let mut sizes = vec![n_inputs];
        sizes.extend_from_slice(layer_sizes);
        let n_layers = layer_sizes.len();

        let layers = (0..n_layers)
            .map(|i| {
                let is_last = i == n_layers - 1;
                Layer::new(sizes[i], sizes[i + 1], !is_last, rng) // ReLU everywhere except the last layer
            })
            .collect();

        MLP { layers }
    }

    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        let mut activations = inputs.to_vec();

        for layer in &self.layers {
            activations = layer.forward(&activations);
        }

        activations
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

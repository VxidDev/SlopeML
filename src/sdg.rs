use crate::value::Value;

pub struct SDG {
    params: Vec<Value>,
    learning_rate: f64,
    momentum: f64,
    velocity: Vec<f64>,
}

impl SDG {
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.set_grad(0.0);
        }
    }

    pub fn step(&mut self) {
        for (p, v) in self.params.iter().zip(self.velocity.iter_mut()) {
            *v = self.momentum * *v + p.grad();
            p.set_data(p.data() - self.learning_rate * *v);
        }
    }

    pub fn new(params: Vec<Value>, learning_rate: f64, momentum: f64) -> SDG {
        let velocity = vec![0.0; params.len()];

        SDG {
            params,
            learning_rate,
            momentum,
            velocity,
        }
    }
}

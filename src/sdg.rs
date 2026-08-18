use crate::value::Value;

pub struct SDG {
    params: Vec<Value>,
    learning_rate: f64,
}

impl SDG {
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.set_grad(0.0);
        }
    }

    pub fn step(&self) {
        for p in &self.params {
            p.set_data(p.data() - self.learning_rate * p.grad());
        }
    }

    pub fn new(params: Vec<Value>, learning_rate: f64) -> SDG {
        SDG {
            params,
            learning_rate,
        }
    }
}

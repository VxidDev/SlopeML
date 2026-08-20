use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::option::Option;
use std::rc::Rc;

use std::hash::{Hash, Hasher};
use std::ops::Add;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

use std::fmt;

pub enum ValueOp {
    Leaf,
    Add(Value, Value),
    Mul(Value, Value),
    Relu(Value),
}

pub struct ValueData {
    pub data: Cell<f64>,
    pub grad: Cell<f64>,
    op: ValueOp,
}

pub struct Graph {
    topo: Vec<Value>,
}

#[derive(Clone)]
pub struct Value(pub Rc<ValueData>);

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Rc::as_ptr(&self.0) as usize).hash(state);
    }
}

impl Eq for Value {}

impl Value {
    pub fn new(data: f64) -> Value {
        let vd = ValueData {
            data: Cell::new(data),
            grad: Cell::new(0.0),
            op: ValueOp::Leaf,
        };

        Value(Rc::new(vd))
    }

    pub fn new_op(data: f64, op: ValueOp) -> Value {
        let vd = ValueData {
            data: Cell::new(data),
            grad: Cell::new(0.0),
            op,
        };

        Value(Rc::new(vd))
    }

    pub fn data(&self) -> f64 {
        self.0.data.get()
    }

    pub fn grad(&self) -> f64 {
        self.0.grad.get()
    }

    pub fn set_data(&self, new_data: f64) {
        self.0.data.set(new_data);
    }

    pub fn set_grad(&self, new_grad: f64) {
        self.0.grad.set(new_grad);
    }

    pub fn children(&self) -> Vec<Value> {
        match &self.0.op {
            ValueOp::Leaf => vec![],
            ValueOp::Add(a, b) | ValueOp::Mul(a, b) => vec![a.clone(), b.clone()],
            ValueOp::Relu(a) => vec![a.clone()],
        }
    }

    pub fn recompute(&self) {
        let new_data = match &self.0.op {
            ValueOp::Leaf => return,
            ValueOp::Add(a, b) => a.data() + b.data(),
            ValueOp::Mul(a, b) => a.data() * b.data(),
            ValueOp::Relu(a) => a.data().max(0.0),
        };

        self.0.data.set(new_data)
    }

    pub fn propagate(&self) {
        let g = self.grad();

        match &self.0.op {
            ValueOp::Leaf => {}

            ValueOp::Add(a, b) => {
                a.0.grad.set(a.0.grad.get() + g);
                b.0.grad.set(b.0.grad.get() + g);
            }

            ValueOp::Mul(a, b) => {
                let (ad, bd) = (a.data(), b.data());

                a.0.grad.set(a.0.grad.get() + bd * g);
                b.0.grad.set(b.0.grad.get() + ad * g);
            }

            ValueOp::Relu(a) => {
                let local = if self.data() > 0.0 { 1.0 } else { 0.0 };
                a.0.grad.set(a.0.grad.get() + local * g);
            }
        }
    }

    pub fn relu(&self) -> Value {
        Value::new_op(self.data().max(0.0), ValueOp::Relu(self.clone()))
    }
}

pub fn build_topo(node: &Value, visited: &mut HashSet<*const ValueData>, topo: &mut Vec<Value>) {
    let ptr = Rc::as_ptr(&node.0);

    if visited.insert(ptr) {
        for child in node.children() {
            build_topo(&child, visited, topo);
        }

        topo.push(node.clone());
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={}, grad={})", self.data(), self.grad())
    }
}

impl Graph {
    pub fn build(output: &Value) -> Graph {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        build_topo(output, &mut visited, &mut topo);
        Graph { topo }
    }

    pub fn forward(&self) {
        for node in &self.topo {
            node.recompute();
        }
    }

    pub fn zero_grad(&self) {
        for node in &self.topo {
            node.set_grad(0.0);
        }
    }

    pub fn backward(&self) {
        self.topo.last().unwrap().set_grad(1.0);
        for node in self.topo.iter().rev() {
            node.propagate();
        }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self: Value, other: Value) -> Value {
        let out_data = self.data() + other.data();
        Value::new_op(out_data, ValueOp::Add(self, other))
    }
}

impl Add for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        let out_data = self.data() + other.data();
        Value::new_op(out_data, ValueOp::Add(self.clone(), other.clone()))
    }
}

impl Add<&Value> for Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        self + other.clone()
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self: Value, other: Value) -> Value {
        let out_data = self.data() * other.data();
        Value::new_op(out_data, ValueOp::Mul(self, other))
    }
}

impl Mul for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        let out_data = self.data() * other.data();
        Value::new_op(out_data, ValueOp::Mul(self.clone(), other.clone()))
    }
}

impl Mul<&Value> for Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        self * other.clone()
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self: Value) -> Value {
        self * Value::new(-1.0)
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self: Value, other: Value) -> Value {
        self + (-other)
    }
}

impl Sub<&Value> for Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        self + (-other.clone())
    }
}

impl Sub for &Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        self.clone() + (-other.clone())
    }
}

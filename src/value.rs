use std::cell::RefCell;
use std::collections::HashSet;
use std::option::Option;
use std::rc::Rc;

use std::hash::{Hash, Hasher};
use std::ops::Add;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

use std::fmt;

pub struct ValueData {
    pub data: f64,
    pub grad: f64,

    _backward: Option<Box<dyn Fn(Value)>>,
    _prev: HashSet<Value>,
}

#[derive(Clone)]
pub struct Value(pub Rc<RefCell<ValueData>>);

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.0.borrow().data == other.0.borrow().data
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Rc::as_ptr(&self.0) as usize).hash(state);
    }
}

impl Eq for Value {}

fn _build_topo(
    node: &Value,
    visited: &mut HashSet<*const RefCell<ValueData>>,
    topo: &mut Vec<Value>,
) {
    let ptr = Rc::as_ptr(&node.0);

    if !visited.contains(&ptr) {
        visited.insert(ptr);

        for child in node.0.borrow()._prev.iter() {
            _build_topo(child, visited, topo);
        }

        topo.push(node.clone());
    }
}

impl Value {
    pub fn new(data: f64) -> Value {
        let vd = ValueData {
            data: data,
            grad: 0.0,
            _backward: None,
            _prev: HashSet::new(),
        };

        Value(Rc::new(RefCell::new(vd)))
    }

    pub fn new_with_children(data: f64, children: Vec<Value>) -> Value {
        let vd = ValueData {
            data: data,
            grad: 0.0,
            _backward: None,
            _prev: children.into_iter().collect(),
        };

        Value(Rc::new(RefCell::new(vd)))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_data(&self, new_data: f64) {
        self.0.borrow_mut().data = new_data;
    }

    pub fn set_grad(&self, new_grad: f64) {
        self.0.borrow_mut().grad = new_grad;
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        _build_topo(self, &mut visited, &mut topo);

        self.0.borrow_mut().grad = 1.0;

        for v in topo.iter().rev() {
            if let Some(f) = &v.0.borrow()._backward {
                f(v.clone());
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = self.0.borrow();
        write!(f, "Value(data={}, grad={})", val.data, val.grad)
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self: Value, other: Value) -> Value {
        let a = self.clone();
        let b = other.clone();

        let out = Value::new_with_children(
            a.0.borrow().data + b.0.borrow().data,
            vec![self.clone(), other.clone()],
        );

        out.0.borrow_mut()._backward = Some(Box::new(move |out_val: Value| {
            let out_grad = out_val.0.borrow().grad;

            self.0.borrow_mut().grad += out_grad;
            other.0.borrow_mut().grad += out_grad;
        }));

        out
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self: Value, other: Value) -> Value {
        let a = self.clone();
        let b = other.clone();

        let out = Value::new_with_children(
            a.0.borrow().data * b.0.borrow().data,
            vec![self.clone(), other.clone()],
        );

        out.0.borrow_mut()._backward = Some(Box::new(move |out_val: Value| {
            let out_grad = out_val.0.borrow().grad;
            let other_data = other.0.borrow().data;
            let self_data = self.0.borrow().data;

            self.0.borrow_mut().grad += other_data * out_grad;
            other.0.borrow_mut().grad += self_data * out_grad;
        }));

        out
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

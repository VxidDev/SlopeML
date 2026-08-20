mod layer;
mod mlp;
mod neuron;
mod sdg;
mod value;

use rand::RngExt;

use layer::Layer;
use mlp::MLP;
use neuron::Neuron;
use sdg::SDG;
use value::{Graph, Value};

use std::{
    f64::consts::PI,
    io::{self, Write},
};

fn mse_loss(predictions: &[Value], targets: &[f64]) -> Value {
    let mut total = Value::new(0.0);

    for (pred, target) in predictions.iter().zip(targets) {
        let goal = Value::new(*target);
        let error = pred - &goal;
        total = total + &error * &error;
    }

    total * Value::new(1.0 / predictions.len() as f64)
}

fn parse_f64(str: String) -> (f64, bool) {
    let x = match str.parse::<f64>() {
        Ok(x) => x,

        Err(_) => {
            println!("Please enter a valid number.");
            return (0.0, false);
        }
    };

    (x, true)
}

fn main() {
    let mut rng = rand::rng();

    let step = 0.1;
    let mut x = -2.0 * PI;
    let mut points: Vec<f64> = Vec::new();

    while x < 2.0 * PI {
        points.push(x);
        x += step;
    }

    let training_data: Vec<(f64, f64)> = points.iter().map(|&x| (x, x.sin())).collect();

    let mlp = MLP::new(1, &[16, 16, 1], &mut rng);
    let mut optimizer = SDG::new(mlp.parameters(), 0.01, 0.5);

    let input_leaves: Vec<Value> = training_data.iter().map(|_| Value::new(0.0)).collect();
    let preds: Vec<Value> = input_leaves
        .iter()
        .map(|leaf| mlp.forward(&[leaf.clone()])[0].clone())
        .collect();

    let targets: Vec<f64> = training_data.iter().map(|(_, t)| *t).collect();
    let loss = mse_loss(&preds, &targets);
    let graph = Graph::build(&loss);

    let mut epoch = 0;

    while loss.data() > 0.01 && epoch < 100_000 {
        epoch += 1;

        for (leaf, (x, _)) in input_leaves.iter().zip(&training_data) {
            leaf.set_data(*x);
        }

        graph.forward();
        graph.zero_grad();
        graph.backward();

        optimizer.step();

        if epoch % 10 == 0 {
            println!("Epoch {}: loss = {}", epoch, loss.data());
        }
    }

    return;

    loop {
        let mut input = String::new();

        print!("user > ");
        io::stdout().flush().expect("Failed to flush stdout.");

        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read user input.");

        input = input.trim().to_string();

        if input == "exit" {
            break;
        }

        let (x, ok) = parse_f64(input);

        if !ok {
            continue;
        };

        let x_wrapped = ((x % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI);

        let predicted = mlp.forward(&[Value::new(x_wrapped)])[0].clone();
        println!("sin > {} (sin({} == {}))", predicted.data(), x, x.sin());
    }
}

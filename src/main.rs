mod neuron;
mod sdg;
mod value;

use rand::RngExt;

use neuron::Neuron;
use sdg::SDG;
use value::Value;

use std::io::{self, Write};

fn mse_loss(predictions: &[Value], targets: &[f64]) -> Value {
    let mut total = Value::new(0.0);

    for (pred, target) in predictions.iter().zip(targets) {
        let goal = Value::new(*target);
        let error = pred - &goal;
        total = total + &error * &error;
    }

    total
}

fn main() {
    let mut rng = rand::rng();

    let data = vec![
        "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    ];

    let points: Vec<(f64, f64)> = (0..=10).map(|i| (i as f64, i as f64)).collect();

    let neuron = Neuron::new(1, &mut rng);
    let optimizer = SDG::new(neuron.parameters(), 0.001);

    for epoch in 0..10 {
        optimizer.zero_grad();

        let preds: Vec<Value> = points
            .iter()
            .map(|(x, _)| neuron.forward(&[Value::new(*x)]))
            .collect();

        let targets: Vec<f64> = points.iter().map(|(_, t)| *t).collect();
        let loss = mse_loss(&preds, &targets);

        loss.backward();
        optimizer.step();

        if epoch % 50 == 0 {
            println!("Epoch {}: loss = {}", epoch, loss.data());
        }
    }

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

        let num = match input.parse::<f64>() {
            Ok(num) => num,

            Err(_) => {
                println!("Please enter a valid number.");
                continue;
            }
        };

        let predicted = neuron.forward(&[Value::new(num)]);
        let idx = predicted.data().round() as i64;

        if idx < 0 || idx >= data.len() as i64 {
            println!("(out of range - predicted index {})", idx);
            continue;
        }

        println!("number_to_word > {}", data[idx as usize]);
    }
}

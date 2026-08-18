mod sdg;
mod value;

use rand::RngExt;

use sdg::SDG;
use value::Value;

use std::io::{self, Write};

fn predict(weight: &Value, bias: &Value, input_number: f64) -> f64 {
    let input = Value::new(input_number);
    let output = input * weight + bias;
    output.data()
}

fn main() {
    let mut rng = rand::rng();

    let data = vec![
        "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    ];

    let points: Vec<(f64, f64)> = (0..=10).map(|i| (i as f64, i as f64)).collect();

    let weight = Value::new(rng.random_range(-1.0..1.0));
    let bias = Value::new(rng.random_range(-1.0..1.0));

    let optimizer = SDG::new(vec![weight.clone(), bias.clone()], 0.001);

    for epoch in 0..500 {
        optimizer.zero_grad();

        let mut total_loss = Value::new(0.0);

        for (x, target) in points.iter() {
            let input = Value::new(*x);
            let goal = Value::new(*target);

            let output = &input * &weight + &bias;
            let error = goal - output;
            let loss = error.clone() * error.clone();

            total_loss = total_loss + loss;
        }

        total_loss.backward();

        optimizer.step();

        if epoch % 50 == 0 {
            println!("Epoch {}: loss = {}", epoch, total_loss.data());
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

        let predicted = predict(&weight, &bias, num);
        let idx = predicted.round() as i64;

        if idx < 0 || idx >= data.len() as i64 {
            println!("(out of range - predicted index {})", idx);
            continue;
        }

        println!("number_to_word > {}", data[idx as usize]);
    }
}

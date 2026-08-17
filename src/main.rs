mod value;

use value::Value;

fn main() {
    let input = Value::new(1.2);
    let goal = Value::new(3.4);
    let weight = Value::new(0.5);

    let learning_rate = 0.01;

    for step in 0..1250 {
        weight.0.borrow_mut().grad = 0.0;

        let output = input.clone() * weight.clone();
        let error = goal.clone() - output.clone();
        let loss = error.clone() * error.clone();

        loss.backward();

        let grad = weight.0.borrow().grad;
        let new_data = weight.0.borrow().data - learning_rate * grad;
        weight.0.borrow_mut().data = new_data;

        if step % 10 == 0 {
            println!(
                "Step {}: output = {}, loss = {}",
                step,
                output.0.borrow().data,
                loss.0.borrow().data
            );
        }
    }

    println!("Final Weight: {}", weight.0.borrow().data);
}

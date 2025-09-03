use std::io;
mod network;

fn read_input<T: std::str::FromStr>(msg: &str) -> T{
    loop {
        let mut input = String::new();
        println!("{}", msg);
        io::stdin()
            .read_line(&mut input)
            .expect("Should never fail to read input");
        match input.trim().parse::<T>() {
            Ok(val) => return val,
            Err(e) => println!("Error: Invalid Input"),
        }
    }
}
fn main() {
    let learning_rate = 0.1;
}
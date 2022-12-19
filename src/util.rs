use std::time::SystemTime;

pub struct Timer(SystemTime);

pub fn closest_divisor(x : usize, by : usize) -> usize {
    x - (x % by)
}

pub fn closest_divisor_and_rem(x : usize, by : usize) -> (usize, usize) {
    let rem = x % by;
    (x - rem, rem)
}

impl Timer {

    pub fn start() -> Self {
        Self(SystemTime::now())
    }

    pub fn reset(&mut self) {
        self.0 = SystemTime::now();
    }

    pub fn time(&mut self, prefix : &str) {
        let now = SystemTime::now();
        println!("{} {}", prefix, now.duration_since(self.0).unwrap().as_micros() as f32 / 1000.0);
        self.0 = now;
    }

}


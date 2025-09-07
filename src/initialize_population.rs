use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Debug)]
pub enum Representation {
    Binary(Vec<u8>),
    Integer(Vec<i32>),
    IntPerm(Vec<i32>),
    Real(Vec<f64>),
}

pub fn generate_binary(pop_size: usize, dim: usize) -> Vec<Representation> {
    let mut rng = rand::thread_rng();
    (0..pop_size)
        .map(|_| {
            let genes: Vec<u8> = (0..dim)
                .map(|_| rng.gen_range(0..=1))
                .collect();
            Representation::Binary(genes)
        })
        .collect()
}

pub fn generate_integer(pop_size: usize, dim: usize, min: i32, max: i32) -> Vec<Representation> {
    let mut rng = rand::thread_rng();
    (0..pop_size)
        .map(|_| {
            let genes: Vec<i32> = (0..dim)
                .map(|_| rng.gen_range(min..=max))
                .collect();
            Representation::Integer(genes)
        })
        .collect()
}

pub fn generate_intperm(pop_size: usize, dim: usize) -> Vec<Representation> {
    let mut rng = rand::thread_rng();
    (0..pop_size)
        .map(|_| {
            let mut perm: Vec<i32> = (1..=dim as i32).collect();
            perm.shuffle(&mut rng);
            Representation::IntPerm(perm)
        })
        .collect()
}

pub fn generate_real(pop_size: usize, dim: usize, min: f64, max: f64) -> Vec<Representation> {
    let mut rng = rand::thread_rng();
    (0..pop_size)
        .map(|_| {
            let genes: Vec<f64> = (0..dim)
                .map(|_| rng.gen_range(min..=max))
                .collect();
            Representation::Real(genes)
        })
        .collect()
}

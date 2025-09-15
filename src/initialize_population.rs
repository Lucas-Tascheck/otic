#![allow(dead_code)]
use rand::seq::SliceRandom;
use serde_json;
use serde::Deserialize;
use std::fs;
use rand::Rng;

#[derive(Debug)]
pub enum Representation {
    Binary(Vec<u8>),
    Integer(Vec<i32>),
    IntPerm(Vec<i32>),
    Real(Vec<f64>),
}

#[derive(Debug)]
pub enum RepresentationType {
    Binary { dim: usize },
    Integer { dim: usize, min: i32, max: i32 },
    IntPerm { dim: usize },
    Real { dim: usize, min: f64, max: f64 },
}

pub fn generate_population(pop_size: usize, repr: RepresentationType) -> Vec<Representation> {
    let mut rng = rand::thread_rng();

    match repr {
        RepresentationType::Binary { dim } => (0..pop_size)
        .map(|_| {
            let genes: Vec<u8> = (0..dim).map(|_| rng.gen_range(0..=1)).collect();
            Representation::Binary(genes)
            })
            .collect(),

            RepresentationType::Integer { dim, min, max } => (0..pop_size)
            .map(|_| {
                let genes: Vec<i32> = (0..dim).map(|_| rng.gen_range(min..=max)).collect();
                Representation::Integer(genes)
            })
            .collect(),

            RepresentationType::IntPerm { dim } => (0..pop_size)
            .map(|_| {
                let mut perm: Vec<i32> = (1..=dim as i32).collect();
                perm.shuffle(&mut rng);
                Representation::IntPerm(perm)
            })
            .collect(),
            
            RepresentationType::Real { dim, min, max } => (0..pop_size)
            .map(|_| {
                let genes: Vec<f64> = (0..dim).map(|_| rng.gen_range(min..=max)).collect();
                Representation::Real(genes)
            })
            .collect(),
        }
    }

#[derive(Deserialize, Debug)]
pub struct Config {
    pub pop: usize,
    pub dim: usize,
    pub runs: usize,
    pub gens: usize
}

pub fn read_config(path: &str) -> Config {
    let data = fs::read_to_string(path).expect("Falha ao ler arquivo de configuração");
    serde_json::from_str(&data).expect("Falha ao parsear JSON")
}
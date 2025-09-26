#![allow(dead_code)]
use rand::seq::SliceRandom;
use serde_json;
use serde::Deserialize;
use std::fs;
use rand::Rng;
use bitvec::prelude::*; // necessário para BitVec

#[derive(Debug, Clone)]
pub enum Representation {
    Binary(BitVec),     // agora é BitVec
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
                let mut genes: BitVec = BitVec::with_capacity(dim);
                for _ in 0..dim {
                    genes.push(rng.gen_bool(0.5));
                }
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

pub fn print_bits(bits: &BitVec) -> String {
    bits.iter().map(|b| if *b { '1' } else { '0' }).collect()
}

// Converte BitVec binário para decimal (LSB à esquerda)
pub fn bitvec_to_dec(bits: &BitSlice) -> u32 {
    let mut value = 0;
    for (i, bit) in bits.iter().enumerate() {
        if *bit {
            value += 1 << i; // LSB à esquerda
        }
    }
    value
}

// Converte BitVec para valor float no intervalo [min, max]
pub fn gen_to_fen(bits: &BitSlice, min: u32, max: u32, bit_length: u32) -> f64 {
    let min_f = min as f64;
    let max_f = max as f64;
    let val = min_f + ((max_f - min_f) / ((2.0f64).powi(bit_length as i32) - 1.0)) * (bitvec_to_dec(bits) as f64);
    val
}

pub fn gen_to_fen_fl(bits: &BitSlice, min: f64, max: f64, bit_length: i32) -> f64 {
    let num = bitvec_to_dec(bits) as f64;
    min + (num / ((2.0f64).powi(bit_length) - 1.0)) * (max - min)
}

// --- Exemplo de main para teste ---
fn main() {
    let pop = generate_population(5, RepresentationType::Binary { dim: 10 });
    for ind in pop {
        if let Representation::Binary(b) = ind {
            println!("Binário: {:?}, Decimal: {}, Fen: {:.2}", b, bitvec_to_dec(&b), gen_to_fen(&b, 0, 40, 10));
        }
    }
}
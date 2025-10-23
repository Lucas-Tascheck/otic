#![allow(dead_code)]
use rand::seq::SliceRandom;
use serde::Deserialize;
use std::fs;
use rand::Rng;
use rayon::prelude::*;
use bitvec::prelude::*;
use plotters::prelude::{BitMapBackend, ChartBuilder, LineSeries, PathElement, IntoDrawingArea};
use plotters::style::{RED, BLUE, GREEN, WHITE, BLACK};
use plotters::prelude::*;

#[derive(Debug, Clone)]
pub enum Representation {
    Binary(BitVec),
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

#[derive(Clone, Debug)]
pub struct Indiv {
    pub genes: BitVec, 
    pub fitness: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IndivMO {
    pub genes: BitVec,
    pub objectives: Vec<f64>,
}

impl IndivMO {
    pub fn new(genes: BitVec, objectives: Vec<f64>) -> Self {
        Self { genes, objectives }
    }
}


pub fn generate_population(pop_size: usize, repr: RepresentationType) -> Vec<Representation> {
    let mut rng = rand::thread_rng();

    match repr {
        RepresentationType::Binary { dim } => (0..pop_size)
            .map(|_| {
                let genes: BitVec = (0..dim).map(|_| rng.gen_bool(0.5)).collect();
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
    pub gens: usize,
}

pub fn read_config(path: &str) -> Config {
    let data = fs::read_to_string(path).expect("Falha ao ler arquivo de configuração");
    serde_json::from_str(&data).expect("Falha ao parsear JSON")
}

pub fn print_bits(bits: &BitVec) -> String {
    bits.iter().map(|b| if *b { '1' } else { '0' }).collect()
}

pub fn bitvec_to_dec(bits: &BitSlice) -> u32 {
    bits.iter()
        .enumerate()
        .fold(0, |acc, (i, bit)| if *bit { acc + (1 << i) } else { acc })
}

pub fn gen_to_fen(bits: &BitSlice, min: u32, max: u32, bit_length: u32) -> f64 {
    let val = min as f64 + ((max - min) as f64 / ((1 << bit_length) - 1) as f64) * bitvec_to_dec(bits) as f64;
    val
}

pub fn gen_to_fen_fl(bits: &BitSlice, min: f64, max: f64, bit_length: i32) -> f64 {
    let num = bitvec_to_dec(bits) as f64;
    min + (num / ((2.0f64).powi(bit_length) - 1.0)) * (max - min)
}

pub fn roulette_once<'a>(pool: &'a [Indiv], rng: &mut impl Rng) -> &'a Indiv {
    let total_fitness: f64 = pool.iter().map(|ind| ind.fitness).sum();
    let r: f64 = rng.r#gen();
    let mut cumulative = 0.0;

    pool.iter()
        .find(|ind| {
            cumulative += ind.fitness / total_fitness;
            r <= cumulative
        })
        .unwrap_or(&pool[pool.len()-1])
}

pub fn roulette(population: &[Indiv], num_pairs: usize) -> Vec<(&Indiv, &Indiv)> {
    let mut rng = rand::thread_rng();
    let mut selected = Vec::with_capacity(num_pairs);

    for _ in 0..num_pairs {
        let parent1 = roulette_once(population, &mut rng);
        let mut parent2 = roulette_once(population, &mut rng);
        while parent2.genes == parent1.genes {
            parent2 = roulette_once(population, &mut rng);
        }
        selected.push((parent1, parent2));
    }

    selected
}

pub fn apply_crossover(pairs: &[(&Indiv, &Indiv)], crossover_prob: f64) -> Vec<Indiv> {
    let mut rng = rand::thread_rng();
    let mut offspring = Vec::with_capacity(pairs.len() * 2);

    for (p1, p2) in pairs {
        if rng.gen_bool(crossover_prob) {
            let len = p1.genes.len();

            let mut point1 = rng.gen_range(1..len);
            let mut point2 = rng.gen_range(1..len);
            if point1 > point2 {
                std::mem::swap(&mut point1, &mut point2);
            }

            let mut child1 = p1.genes.clone();
            let mut child2 = p2.genes.clone();

            for i in point1..point2 {
                let temp = child1[i];
                child1.set(i, child2[i]);
                child2.set(i, temp);
            }

            offspring.push(Indiv { genes: child1, fitness: 0.0 });
            offspring.push(Indiv { genes: child2, fitness: 0.0 });
        } else {
            offspring.push((*p1).clone());
            offspring.push((*p2).clone());
        }
    }

    offspring
}

pub fn mutation(population: &mut [Indiv], mutation_prob: f64) {
    let mut rng = rand::thread_rng();

    for ind in population.iter_mut() {
        for i in 0..ind.genes.len() {
            if rng.gen_bool(mutation_prob) {
                let current = ind.genes[i];
                ind.genes.set(i, !current);
                //println!("Teve Mutação!");
            }
        }
    }
}


pub fn plot_convergence(best: &[f64], mean: &[f64], worst: &[f64], filename: &str) {
    let gens = best.len();
    let root = BitMapBackend::new(filename, (1000, 700)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Convergência do AG", ("sans-serif", 35))
        .margin(5)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(0..gens, 0.0..1.0)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Gerações")
        .y_desc("Fitness")
        .label_style(("sans-serif", 18))
        .axis_desc_style(("sans-serif", 22))
        .light_line_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();

    // Linhas mais grossas
    chart
    .draw_series(LineSeries::new(
        best.iter().enumerate().map(|(i, &v)| (i, v)),
        RED.mix(0.9).stroke_width(3),
    ))
    .unwrap()
    .label("Melhor")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], RED.mix(0.9).stroke_width(3)));

    chart
        .draw_series(LineSeries::new(
            mean.iter().enumerate().map(|(i, &v)| (i, v)),
            BLUE.mix(0.9).stroke_width(3),
        ))
        .unwrap()
        .label("Média")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], BLUE.mix(0.9).stroke_width(3)));

    chart
        .draw_series(LineSeries::new(
            worst.iter().enumerate().map(|(i, &v)| (i, v)),
            GREEN.mix(0.9).stroke_width(3),
        ))
        .unwrap()
        .label("Pior")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], GREEN.mix(0.9).stroke_width(3)));


    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 20))
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .unwrap();
}

pub fn tournament_selection(population: &[Indiv], num_pairs: usize, k: usize) -> Vec<(&Indiv, &Indiv)> {
    let mut rng = rand::thread_rng();
    let mut selected = Vec::with_capacity(num_pairs);

    for _ in 0..num_pairs {
        // Seleciona pai 1
        let mut candidates: Vec<&Indiv> = population.choose_multiple(&mut rng, k).collect();
        candidates.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        let parent1 = candidates[0];

        // Seleciona pai 2 (garante diferente)
        let mut parent2;
        loop {
            let mut candidates: Vec<&Indiv> = population.choose_multiple(&mut rng, k).collect();
            candidates.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            parent2 = candidates[0];
            if parent2.genes != parent1.genes {
                break;
            }
        }

        selected.push((parent1, parent2));
    }

    selected
}

pub fn uniform_crossover(pairs: &[(&Indiv, &Indiv)], crossover_prob: f64) -> Vec<Indiv> {
    let mut rng = rand::thread_rng();
    let mut offspring = Vec::with_capacity(pairs.len() * 2);

    for (p1, p2) in pairs {
        if rng.gen_bool(crossover_prob) {
            let len = p1.genes.len();
            let mut child1 = p1.genes.clone();
            let mut child2 = p2.genes.clone();

            for i in 0..len {
                if rng.gen_bool(0.5) {
                    child1.set(i, p2.genes[i]);
                    child2.set(i, p1.genes[i]);
                }
            }

            offspring.push(Indiv { genes: child1, fitness: 0.0 });
            offspring.push(Indiv { genes: child2, fitness: 0.0 });
        } else {
            offspring.push((*p1).clone());
            offspring.push((*p2).clone());
        }
    }

    offspring
}


pub fn plot_population_scatter(pop: &[IndivMO], filename: &str) {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let x_min = pop.iter().map(|p| p.objectives[0]).fold(f64::INFINITY, f64::min);
    let x_max = pop.iter().map(|p| p.objectives[0]).fold(f64::NEG_INFINITY, f64::max);
    let y_min = pop.iter().map(|p| p.objectives[1]).fold(f64::INFINITY, f64::min);
    let y_max = pop.iter().map(|p| p.objectives[1]).fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("População Inicial - ZDT1 Discreta", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("f1")
        .y_desc("f2")
        .label_style(("sans-serif", 16))
        .axis_desc_style(("sans-serif", 18))
        .draw()
        .unwrap();

    chart
        .draw_series(
            pop.iter().map(|ind| Circle::new((ind.objectives[0], ind.objectives[1]), 3, BLUE.filled())),
        )
        .unwrap();
}
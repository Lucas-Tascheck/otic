use crate::functions::{
    Representation, RepresentationType, generate_population, gen_to_fen_fl
};
use rayon::prelude::*;
use rand::Rng;

#[derive(Clone, Debug)]
struct Indiv {
    genes: String,
    fitness: f64,
}

fn roulette_selection_without_replacement(
    population: &Vec<Indiv>,
    num_to_select: usize,
) -> Vec<Indiv> {
    let mut selected = Vec::new();
    let mut pool = population.clone();

    for _ in 0..num_to_select {
        let total_fitness: f64 = pool.iter().map(|ind| ind.fitness).sum();

        let mut cumulative = 0.0;
        let mut wheel = Vec::new();
        for ind in &pool {
            cumulative += ind.fitness / total_fitness;
            wheel.push((cumulative, ind.clone()));
        }

        let r: f64 = rand::thread_rng().r#gen::<f64>();

        if let Some((_, chosen)) = wheel.into_iter().find(|(prob, _)| r <= *prob) {
            selected.push(chosen.clone());
            pool.retain(|x| x.genes != chosen.genes);
        }
    }

    selected
}

pub fn evaluate_individual(ind: &Representation, minimize: bool) -> f64 {
    match ind {
        Representation::Binary(genes) => {
            let x = gen_to_fen_fl(&genes, -2.0, 2.0, 16) as f64;
            let normalize = 6.0;

            let fx = (20.0f64 * x).cos() - x.abs() / 2.0f64 + (x.powi(3) / 4.0f64);

            if minimize {
                let c_max: f64 = 2.0; 
                ((c_max - fx)/normalize).max(0.0)
            } else {
                let c_min: f64 = 4.0;
                ((fx + c_min)/normalize).max(0.0)
            }
        }
        _ => panic!("Representação inválida para este exercício, use Real."),
    }
}

pub fn run_exercicio1(pop: usize, dim: usize, gens: usize, runs: usize) {
    println!("=== EX 1: Maximização e Minimização de Função Algébrica ===");

    (1..=runs).into_par_iter().for_each(|run| {
        let mut global_best_score = 0.0;
        let mut global_best_genes: Option<String> = None;

        let mut global_worst_score = f64::MIN;
        let mut global_worst_genes: Option<String> = None;

        for _g in 1..=gens {
            let population = generate_population(
                pop,
                RepresentationType::Binary { dim }
            );

            for ind in &population {
                let score_max = evaluate_individual(ind, false);
                let score_min = evaluate_individual(ind, true);

                if score_max > global_best_score {
                    global_best_score = score_max;
                    if let Representation::Binary(genes) = ind {
                        global_best_genes = Some(genes.clone());
                    }
                }

                if score_min > global_worst_score {
                    global_worst_score = score_min;
                    if let Representation::Binary(genes) = ind {
                        global_worst_genes = Some(genes.clone());
                    }
                }
            }

            let evaluated: Vec<Indiv> = population.iter().map(|ind| {
                let fitness = evaluate_individual(ind, false); 
                let genes = if let Representation::Binary(g) = ind {
                    g.clone()
                } else {
                    panic!("Representação inválida");
                };
                Indiv { genes, fitness }
            }).collect();

            let selecionados = roulette_selection_without_replacement(&evaluated, evaluated.len());
            //println!("Indivíduos selecionados (roleta): {:?}", selecionados);
        }

        if let Some(best_genes) = global_best_genes {
            println!(
                "Run {} -> Maximização: genes = {:?} | Valor de x = {:?} | fitness = {:.4}",
                run, best_genes, gen_to_fen_fl(&best_genes, -2.0, 2.0, 16), global_best_score
            );
        }

        if let Some(worst_genes) = global_worst_genes {
            println!(
                "Run {} -> Minimização: genes = {:?} | Valor de x = {:?} | fitness = {:.4}\n",
                run, worst_genes, gen_to_fen_fl(&worst_genes, -2.0, 2.0, 16), global_worst_score
            );
        }
    });
}

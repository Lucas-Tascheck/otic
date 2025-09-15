use crate::initialize_population::{
    Representation, RepresentationType, generate_population
};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Clause {
    pub vars: [i32; 3], // positivo = normal, negativo = negado
}

#[derive(Debug)]
pub struct SATInstance {
    pub clauses: Vec<Clause>,
    pub n_vars: usize,
}

pub fn toy_instance() -> SATInstance {
    SATInstance {
        n_vars: 4,
        clauses: vec![
            Clause { vars: [1, -2, 3] },   // (x1 ∨ ¬x2 ∨ x3)
            Clause { vars: [-1, 2, 4] },   // (¬x1 ∨ x2 ∨ x4)
            Clause { vars: [-3, -4, 2] },  // (¬x3 ∨ ¬x4 ∨ x2)
        ],
    }
}

pub fn evaluate(instance: &SATInstance, assignment: &Vec<u8>) -> i32 {
    let mut satisfied = 0;
    for clause in &instance.clauses {
        let mut clause_sat = false;
        for &var in &clause.vars {
            let idx = var.abs() as usize - 1;
            let val = assignment[idx] == 1;
            if (var > 0 && val) || (var < 0 && !val) {
                clause_sat = true;
                break;
            }
        }
        if clause_sat {
            satisfied += 1;
        }
    }
    satisfied
}

pub fn run_3sat(pop: usize, dim: usize, gens: usize, runs: usize) {
    println!("\n=== EX 2: Problema 3-SAT ===");

    (1..=runs).into_par_iter().for_each(|run| {
        let instance = toy_instance();
        let mut global_best_score = -1;
        let mut global_best_ind: Vec<u8> = Vec::new();

        for g in 1..=gens {
            let population = generate_population(pop, RepresentationType::Binary { dim: instance.n_vars });

            let mut best_score = -1;
            let mut best_ind: Vec<u8> = Vec::new();

            for ind in &population {
                if let Representation::Binary(genes) = ind {
                    let score = evaluate(&instance, genes);
                    if score > best_score {
                        best_score = score;
                        best_ind = genes.clone();
                    }
                }
            }

            if best_score > global_best_score {
                global_best_score = best_score;
                global_best_ind = best_ind.clone();
            }
        }
        println!(
            "Run {} concluída -> Melhor indivíduo global: {:?} | Score = {}",
            run, global_best_ind, global_best_score
        );
    });
}


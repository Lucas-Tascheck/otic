use crate::utils::functions::*;
use rayon::prelude::*;
use bitvec::prelude::*;

#[derive(Debug, Clone)]
pub struct Clause {
    pub vars: [i32; 3],
}

#[derive(Debug, Clone)]
pub struct SATInstance {
    pub clauses: Vec<Clause>,
    pub n_vars: usize,
}

pub fn toy_instance() -> SATInstance {
    SATInstance {
        n_vars: 4,
        clauses: vec![
            Clause { vars: [1, -2, 3] },
            Clause { vars: [-1, 2, 4] },
            Clause { vars: [-3, -4, 2] },
        ],
    }
}

pub fn evaluate(instance: &SATInstance, assignment: &BitVec) -> i32 {
    let mut satisfied = 0;
    for clause in &instance.clauses {
        let mut clause_sat = false;
        for &var in &clause.vars {
            let idx = var.abs() as usize - 1;
            let val = assignment[idx];
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

pub fn run_3sat(pop: usize, _dim: usize, gens: usize, runs: usize) {
    println!("\n=== EX 3: Problema 3-SAT ===");

    (1..=runs).into_par_iter().for_each(|run| {
        let instance = toy_instance();
        let mut global_best_score = -1;
        let mut global_best_ind: BitVec = BitVec::new();

        for _g in 1..=gens {
            let population = generate_population(pop, RepresentationType::Binary { dim: instance.n_vars });

            for ind in &population {
                if let Representation::Binary(genes) = ind {
                    let score = evaluate(&instance, genes);
                    if score > global_best_score {
                        global_best_score = score;
                        global_best_ind = genes.clone();
                    }
                }
            }
        }

        println!(
            "Run {} concluída -> Melhor indivíduo global: {:?} | Score = {}",
            run, global_best_ind, global_best_score
        );
    });
}
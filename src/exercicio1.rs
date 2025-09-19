use crate::initialize_population::{
    Representation, RepresentationType, generate_population
};
use rayon::prelude::*;

pub fn evaluate_individual(ind: &Representation, minimize: bool) -> f64 {
    match ind {
        Representation::Real(genes) => {
            let x = genes[0];
            let normalize = 6.0;

            let fx = (20.0 * x).cos() - x.abs() / 2.0 + (x.powi(3) / 4.0);

            if minimize {
                let c_max = 2.0; 
                ((c_max - fx)/normalize).max(0.0)
            } else {
                let c_min = 4.0;
                ((fx + c_min)/normalize).max(0.0)
            }
        }
        _ => panic!("Representação inválida para este exercício, use Real."),
    }
}

pub fn run_exercicio1(pop: usize, dim: usize, gens: usize, runs: usize) {
    println!("=== EX 1: Maximização e Minimização de Função Algébrica ===");

    (1..=runs).into_par_iter().for_each(|run| {
        let mut global_best_score = f64::MIN;
        let mut global_best_genes: Option<Vec<f64>> = None;

        let mut global_worst_score = f64::MIN;
        let mut global_worst_genes: Option<Vec<f64>> = None;

        for _g in 1..=gens {
            let population = generate_population(
                pop,
                RepresentationType::Real { dim, min: -2.0, max: 2.0 }
            );

            for ind in &population {
                let score_max = evaluate_individual(ind, false);
                let score_min = evaluate_individual(ind, true);

                if score_max > global_best_score {
                    global_best_score = score_max;
                    if let Representation::Real(genes) = ind {
                        global_best_genes = Some(genes.clone());
                    }
                }

                if score_min > global_worst_score {
                    global_worst_score = score_min;
                    if let Representation::Real(genes) = ind {
                        global_worst_genes = Some(genes.clone());
                    }
                }
            }
        }

        if let Some(best_genes) = global_best_genes {
            println!(
                "Run {} -> Maximização: genes = {:?} | fitness = {:.4}",
                run, best_genes, global_best_score
            );
        }

        if let Some(worst_genes) = global_worst_genes {
            println!(
                "Run {} -> Minimização: genes = {:?} | fitness = {:.4}\n",
                run, worst_genes, global_worst_score
            );
        }
    });
}

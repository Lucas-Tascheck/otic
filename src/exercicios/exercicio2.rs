use crate::utils::functions::*;
use rayon::prelude::*;
use bitvec::prelude::BitVec;

fn objective_fn(st: f64, lx: f64) -> f64 {
    if st + lx == 0.0 {
        return 0.0;
    }
    (30.0 * st + 40.0 * lx) / 1360.0
}

fn penalty(st: f64, lx: f64) -> f64 {
    ((st + 2.0 * lx - 40.0) / 16.0).max(0.0)
}

fn evaluate_individual(ind: &Representation, minimize: bool, k: f64) -> f64 {
    match ind {
        Representation::Binary(genes) => {
            let mid = genes.len() / 2;
            let bits_st = &genes[..mid];
            let bits_lx = &genes[mid..];

            let st = gen_to_fen(bits_st, 0, 24, 5);
            let lx = gen_to_fen(bits_lx, 0, 16, 5);

            (objective_fn(st, lx) - k * penalty(st, lx)).max(0.0)
        }
        _ => panic!("Representação inválida para este exercício, use Binary."),
    }
}

pub fn lucro_radios(genes: &BitVec) -> f64 {
    let mid = genes.len() / 2;
    let bits_st = &genes[..mid];
    let bits_lx = &genes[mid..];

    let st = gen_to_fen(bits_st, 0, 24, 5).round();
    let lx = gen_to_fen(bits_lx, 0, 16, 5).round();
    objective_fn(st, lx) * 1360.0
}

pub fn run_exercicio2(pop: usize, dim: usize, gens: usize, runs: usize, k: f64) {
    println!("=== EX 2: Maximização e Minimização do Problema Dos Rádios ===");
    
    (1..=runs).into_par_iter().for_each(|run| {
        let mut global_best_score = 0.0;
        let mut global_best_genes: Option<BitVec> = None;

        for _g in 1..=gens {
            let population = generate_population(pop, RepresentationType::Binary { dim });

            for ind in &population {
                let score = evaluate_individual(ind, false, k);

                if score >= global_best_score {
                    global_best_score = score;
                    if let Representation::Binary(genes) = ind {
                        global_best_genes = Some(genes.clone());
                    }
                }
            }
        }

        if let Some(best_genes) = &global_best_genes {
            println!(
                "Run {} -> Maximização: genes = {:?} | fitness = {:.4} | Lucro = {:.2}",
                run, print_bits(&best_genes), global_best_score, lucro_radios(best_genes)
            );
        }
    });
}

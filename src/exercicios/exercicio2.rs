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

fn evaluate_individual(ind: &Representation, k: f64) -> f64 {
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

pub fn run_exercicio2(pop: usize, dim: usize, gens: usize, runs: usize, crossover_prob: f64, mutation_prob: f64, k: f64) {
    println!("=== EX 2: Maximização e Minimização do Problema Dos Rádios ===");
    
    (1..=runs).into_par_iter().for_each(|run| {
        let mut global_best_score = 0.0;
        let mut global_best_genes: Option<BitVec> = None;
        let mut population = generate_population(pop, RepresentationType::Binary { dim });

        for _g in 1..=gens {
            let evaluated: Vec<Indiv> = population.par_iter().map(|ind| {
                let fitness = match ind {
                    Representation::Binary(genes) => evaluate_individual(&Representation::Binary(genes.clone()), 1.0),
                    _ => panic!("Representação inválida"),
                };
                if let Representation::Binary(genes) = ind {
                    Indiv { genes: genes.clone(), fitness }
                } else { unreachable!() }
            }).collect();

            for ind in &evaluated {
                if ind.fitness > global_best_score {
                    global_best_score = ind.fitness;
                    global_best_genes = Some(ind.genes.clone());
                }
            }

            let mean_fitness: f64 = evaluated.par_iter().map(|ind| ind.fitness).sum::<f64>() / evaluated.len() as f64;

            let selecionados = roulette(&evaluated, evaluated.len() / 2);
            let mut crossover = apply_crossover(&selecionados, crossover_prob);

            mutation(&mut crossover, mutation_prob);

            population = crossover
                .into_iter()
                .map(|ind| Representation::Binary(ind.genes))
                .collect();

            println!("Geração -> Max fitness: {:.4}, Média: {:.4}", global_best_score, mean_fitness);
        }
        for (i, ind) in population.iter().enumerate() {
            if let Representation::Binary(genes) = ind {
                println!("Ind {} -> {}", i + 1, print_bits(genes));
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

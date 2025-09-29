use crate::utils::functions::*;
use rayon::prelude::*;
use rand::Rng;
use bitvec::prelude::BitVec;

pub fn evaluate_individual(ind: &Representation, minimize: bool) -> f64 {
    match ind {
        Representation::Binary(genes) => {
            let x = gen_to_fen_fl(&genes, -2.0, 2.0, 16);
            let normalize = 6.0;
            let fx = (20.0 * x).cos() - x.abs() / 2.0 + (x.powi(3) / 4.0);

            if minimize {
                let c_max: f64 = 2.0;
                ((c_max - fx) / normalize).max(0.0)
            } else {
                let c_min: f64 = 4.0;
                ((fx + c_min) / normalize).max(0.0)
            }
        }
        _ => panic!("Representação inválida, use Binary."),
    }
}

pub fn run_exercicio1(pop: usize, dim: usize, gens: usize, runs: usize, crossover_prob: f64, mutation_prob: f64, maximize: bool) {
    println!("=== EX 1: Função Algébrica Otimizada ===");

    (1..=runs).into_par_iter().for_each(|run| {
        let mut global_best_score = if maximize { f64::MIN } else { f64::MAX };
        let mut global_best_genes: Option<BitVec> = None;

        let mut population: Vec<Representation> = generate_population(pop, RepresentationType::Binary { dim });

        for _ in 0..gens {
            let evaluated: Vec<Indiv> = population.par_iter().map(|ind| {
                let fitness = match ind {
                    Representation::Binary(genes) => evaluate_individual(&Representation::Binary(genes.clone()), !maximize),
                    _ => panic!("Representação inválida"),
                };
                if let Representation::Binary(genes) = ind {
                    Indiv { genes: genes.clone(), fitness }
                } else { unreachable!() }
            }).collect();

            for ind in &evaluated {
                if maximize && ind.fitness > global_best_score {
                    global_best_score = ind.fitness;
                    global_best_genes = Some(ind.genes.clone());
                } else if !maximize && ind.fitness < global_best_score {
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

            // println!("Geração -> Max fitness: {:.4}, Média: {:.4}", global_best_score, mean_fitness);
        }

        if let Some(best_genes) = global_best_genes {
            println!(
                "Run {} -> Melhor indivíduo: genes = {} | Valor de x = {:.4} | fitness = {:.4}",
                run, print_bits(&best_genes), gen_to_fen_fl(&best_genes, -2.0, 2.0, 16), global_best_score
            );
        }
    });
}
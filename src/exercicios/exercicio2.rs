use crate::utils::functions::*;
use rayon::prelude::*;
use bitvec::prelude::BitVec;
use std::sync::Mutex;
use csv::Writer;

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
    println!("=== EX 2: Problema Dos Rádios (com boxplot) ===");

    let writer = Mutex::new(Writer::from_path("boxplot_radios.csv").unwrap());

    (1..=runs).into_par_iter().for_each(|run| {
        let mut global_best_score = 0.0;
        let mut global_best_genes: Option<BitVec> = None;
        let mut population = generate_population(pop, RepresentationType::Binary { dim });

        let mut best_per_gen = Vec::with_capacity(gens);
        let mut mean_per_gen = Vec::with_capacity(gens);
        let mut worst_per_gen = Vec::with_capacity(gens);

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

            let best = evaluated.iter().map(|ind| ind.fitness).fold(f64::MIN, f64::max);
            let worst = evaluated.iter().map(|ind| ind.fitness).fold(f64::MAX, f64::min);
            let mean = evaluated.iter().map(|ind| ind.fitness).sum::<f64>() / evaluated.len() as f64;

            best_per_gen.push(best);
            mean_per_gen.push(mean);
            worst_per_gen.push(worst);

            // seleção, crossover e mutação
            let selecionados = tournament_selection(&evaluated, evaluated.len() / 2, 3);
            let mut crossover = apply_crossover(&selecionados, crossover_prob);
            mutation(&mut crossover, mutation_prob);

            population = crossover
                .into_iter()
                .map(|ind| Representation::Binary(ind.genes))
                .collect();
        }

        // salva resultados para boxplot
        let mut w = writer.lock().unwrap();
        for (generation, best) in best_per_gen.iter().enumerate() {
            w.write_record(&[
                run.to_string(),
                generation.to_string(),
                best.to_string(),
            ]).unwrap();
        }

        // gera gráfico de convergência
        let filename = format!("convergencia_radios_run{}.png", run);
        plot_convergence(&best_per_gen, &mean_per_gen, &worst_per_gen, &filename);

        if let Some(best_genes) = &global_best_genes {
            println!(
                "Run {} -> Melhor indivíduo: {:?} | fitness = {:.4} | Lucro = {:.2}",
                run,
                print_bits(&best_genes),
                global_best_score,
                lucro_radios(best_genes)
            );
        }
    });
}
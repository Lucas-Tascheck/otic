use crate::utils::functions::*; // Importa tudo do seu utilitário
use rayon::prelude::*;
use std::sync::Mutex;
use csv::Writer;
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::collections::HashSet; // Para o crossover OX1

#[derive(Clone, Debug)]
struct IndivPerm {
    pub genes: Vec<i32>,
    pub fitness: f64,
}

fn evaluate_nqueens(genes: &Vec<i32>) -> f64 {
    let n = genes.len();
    if n < 2 { return 0.0; }

    let mut attacks = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let row_i = genes[i] as i32;
            let row_j = genes[j] as i32;
            let col_i = i as i32;
            let col_j = j as i32;

            if (row_i - row_j).abs() == (col_i - col_j).abs() {
                attacks += 1;
            }
        }
    }

    let max_non_attacking_pairs = (n * (n - 1)) / 2;
    (max_non_attacking_pairs as f64) - (attacks as f64)
}


fn tournament_selection_perm(population: &[IndivPerm], num_pairs: usize, k: usize) -> Vec<(&IndivPerm, &IndivPerm)> {
    let mut rng = rand::thread_rng();
    let mut selected = Vec::with_capacity(num_pairs);

    for _ in 0..num_pairs {
        let mut candidates: Vec<&IndivPerm> = population.choose_multiple(&mut rng, k).collect();
        candidates.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        let parent1 = candidates[0];

        let mut parent2;
        loop {
            let mut candidates: Vec<&IndivPerm> = population.choose_multiple(&mut rng, k).collect();
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

fn ox1_crossover(p1: &Vec<i32>, p2: &Vec<i32>) -> (Vec<i32>, Vec<i32>) {
    let n = p1.len();
    let mut rng = thread_rng();
    let (mut cut1, mut cut2) = (rng.gen_range(0..n), rng.gen_range(0..n));
    if cut1 > cut2 { std::mem::swap(&mut cut1, &mut cut2); }

    let mut c1 = vec![-1; n];
    let mut c2 = vec![-1; n];

    c1[cut1..=cut2].copy_from_slice(&p1[cut1..=cut2]);
    c2[cut1..=cut2].copy_from_slice(&p2[cut1..=cut2]);

    let slice1: HashSet<i32> = c1[cut1..=cut2].iter().cloned().collect();
    let slice2: HashSet<i32> = c2[cut1..=cut2].iter().cloned().collect();

    let mut p2_idx = (cut2 + 1) % n;
    let mut c1_idx = (cut2 + 1) % n;
    while c1_idx != cut1 {
        if !slice1.contains(&p2[p2_idx]) {
            c1[c1_idx] = p2[p2_idx];
            c1_idx = (c1_idx + 1) % n;
        }
        p2_idx = (p2_idx + 1) % n;
    }

    let mut p1_idx = (cut2 + 1) % n;
    let mut c2_idx = (cut2 + 1) % n;
    while c2_idx != cut1 {
        if !slice2.contains(&p1[p1_idx]) {
            c2[c2_idx] = p1[p1_idx];
            c2_idx = (c2_idx + 1) % n;
        }
        p1_idx = (p1_idx + 1) % n;
    }
    
    (c1, c2)
}

fn apply_crossover_perm<'a>(pairs: &[(&'a IndivPerm, &'a IndivPerm)], crossover_prob: f64) -> Vec<IndivPerm> {
    let mut rng = rand::thread_rng();
    let mut offspring = Vec::with_capacity(pairs.len() * 2);

    for (p1, p2) in pairs {
        if rng.gen_bool(crossover_prob) {
            let (genes1, genes2) = ox1_crossover(&p1.genes, &p2.genes);
            offspring.push(IndivPerm { genes: genes1, fitness: 0.0 });
            offspring.push(IndivPerm { genes: genes2, fitness: 0.0 });
        } else {
            offspring.push((*p1).clone());
            offspring.push((*p2).clone());
        }
    }
    offspring
}

fn mutation_perm(population: &mut [IndivPerm], mutation_prob: f64) {
    let mut rng = rand::thread_rng();
    let n = population[0].genes.len();
    if n == 0 { return; }

    for ind in population.iter_mut() {
        for i in 0..n {
            if rng.gen_bool(mutation_prob) {
                let j = rng.gen_range(0..n);
                ind.genes.swap(i, j);
            }
        }
    }
}

fn print_board(genes: &Vec<i32>) {
    let n = genes.len();
    println!("--- Solução N-Rainhas (N={}) ---", n);
    for &row_pos in genes {
        let mut line = String::new();
        for j in 0..n {
            if j == (row_pos as usize) {
                line.push_str(" Q ");
            } else {
                line.push_str(" . ");
            }
        }
        println!("{}", line);
    }
    println!("---------------------------------");
}

pub fn run_nqueens(n_queens: usize, pop: usize, gens: usize, runs: usize, crossover_prob: f64, mutation_prob: f64) {
    println!("=== Problema N-Rainhas (N={}) ===", n_queens);

    let writer = Mutex::new(Writer::from_path("boxplot_nqueens.csv").unwrap());
    
    let perfect_fitness = (n_queens * (n_queens - 1)) as f64 / 2.0;

    (1..=runs).into_par_iter().for_each(|run| {
        let mut global_best_score = -1.0;
        let mut global_best_genes: Option<Vec<i32>> = None;

        let evaluate_nqueens_adapted = |genes: &Vec<i32>| {
            let genes_0_indexed: Vec<i32> = genes.iter().map(|&g| g - 1).collect();
            evaluate_nqueens(&genes_0_indexed)
        };

        let mut population = generate_population(pop, RepresentationType::IntPerm { dim: n_queens });

        let mut best_per_gen = Vec::with_capacity(gens);
        let mut mean_per_gen = Vec::with_capacity(gens);
        let mut worst_per_gen = Vec::with_capacity(gens);

        for r#gen in 1..=gens {
            let evaluated: Vec<IndivPerm> = population.par_iter().map(|ind| {
                if let Representation::IntPerm(genes) = ind {
                    let fitness = evaluate_nqueens_adapted(genes);
                    IndivPerm { genes: genes.clone(), fitness }
                } else { panic!("Representação inválida, use IntPerm."); }
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

            if (global_best_score - perfect_fitness).abs() < f64::EPSILON {
                println!("Run {} encontrou solução perfeita na Geração {}!", run, r#gen);
                let final_best = best_per_gen.clone();
                let final_mean = mean_per_gen.clone();
                let final_worst = worst_per_gen.clone();
                for _g in r#gen..=gens {
                    best_per_gen.push(final_best.last().unwrap().clone());
                    mean_per_gen.push(final_mean.last().unwrap().clone());
                    worst_per_gen.push(final_worst.last().unwrap().clone());
                }
                break; 
            }

            let selecionados = tournament_selection_perm(&evaluated, evaluated.len(), 3);

            let mut crossover = apply_crossover_perm(&selecionados, crossover_prob);

            mutation_perm(&mut crossover, mutation_prob);

            population = crossover
                .into_iter()
                .map(|ind| Representation::IntPerm(ind.genes))
                .collect();
        }

        let mut w = writer.lock().unwrap();
        for (generation, best) in best_per_gen.iter().enumerate() {
            w.write_record(&[
                run.to_string(),
                (generation + 1).to_string(), 
                best.to_string(),
            ]).unwrap();
        }
        
        let filename = format!("convergencia_nqueens_run{}.png", run);
        let plot_best: Vec<f64> = best_per_gen.iter().map(|&f| f / perfect_fitness).collect();
        let plot_mean: Vec<f64> = mean_per_gen.iter().map(|&f| f / perfect_fitness).collect();
        let plot_worst: Vec<f64> = worst_per_gen.iter().map(|&f| f / perfect_fitness).collect();
        
        plot_convergence(&plot_best, &plot_mean, &plot_worst, &filename);


        if let Some(best_genes) = &global_best_genes {
            let attacks = perfect_fitness - global_best_score;
            println!(
                "Run {} -> Melhor fitness = {:.0} (Perfeito é {:.0}) | Ataques = {:.0}",
                run,
                global_best_score,
                perfect_fitness,
                attacks
            );
            if attacks == 0.0 {
                let genes_0_indexed: Vec<i32> = best_genes.iter().map(|&g| g - 1).collect();
                print_board(&genes_0_indexed);
            }
        }
    });
}
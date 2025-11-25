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

fn roulette_wheel_selection(population: &[IndivPerm], num_pairs: usize) -> Vec<(&IndivPerm, &IndivPerm)> {
    let mut rng = rand::thread_rng();
    let total_fitness: f64 = population.iter().map(|ind| ind.fitness).sum();
    let mut selected = Vec::with_capacity(num_pairs);

    for _ in 0..num_pairs {
        let parent1 = select_one_by_roulette(population, total_fitness, &mut rng);
        let mut parent2;
        loop {
            parent2 = select_one_by_roulette(population, total_fitness, &mut rng);
            if parent2.genes != parent1.genes {
                break;
            }
        }
        selected.push((parent1, parent2));
    }
    selected
}

fn select_one_by_roulette<'a>(
    population: &'a [IndivPerm],
    total_fitness: f64,
    rng: &mut impl Rng,
) -> &'a IndivPerm {
    let mut cumulative_probability = 0.0;
    let random_value = rng.gen_range(0.0..total_fitness);

    for individual in population {
        cumulative_probability += individual.fitness;
        if cumulative_probability >= random_value {
            return individual;
        }
    }
    &population[population.len() - 1] // Retorna o último indivíduo como fallback
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
        // Apply mutation per individual, not per gene to reduce disruption
        if rng.gen_bool(mutation_prob) {
            // Swap two random positions
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            if i != j {
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

fn tournament_selection(population: &[IndivPerm], num_pairs: usize, tournament_size: usize) -> Vec<(&IndivPerm, &IndivPerm)> {
    let mut rng = rand::thread_rng();
    let mut selected = Vec::with_capacity(num_pairs);

    for _ in 0..num_pairs {
        let parent1 = select_one_by_tournament(population, tournament_size, &mut rng);
        let parent2 = select_one_by_tournament(population, tournament_size, &mut rng);
        selected.push((parent1, parent2));
    }
    selected
}

fn select_one_by_tournament<'a>(
    population: &'a [IndivPerm],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> &'a IndivPerm {
    let mut best: Option<&IndivPerm> = None;
    for _ in 0..tournament_size {
        let candidate = &population[rng.gen_range(0..population.len())];
        if best.is_none() || candidate.fitness > best.unwrap().fitness {
            best = Some(candidate);
        }
    }
    best.unwrap()
}

pub fn run_nqueens(n_queens: usize, pop: usize, gens: usize, runs: usize, crossover_prob: f64, mutation_prob: f64, max_attempts: usize) {
    println!("=== Problema N-Rainhas (N={}) ===", n_queens);

    let csv_filename = format!("boxplot_nqueens_n{}_cx{}_mut{}.csv", n_queens, crossover_prob, mutation_prob);
    let writer = Mutex::new({
        let mut wtr = Writer::from_path(&csv_filename).unwrap();
        wtr.write_record(&["Run", "Generation", "Best_Fitness", &format!("N={}", n_queens), &format!("Crossover={}", crossover_prob), &format!("Mutation={}", mutation_prob)]).unwrap();
        wtr
    });
    
    let perfect_fitness = (n_queens * (n_queens - 1)) as f64 / 2.0;

    (1..=runs).into_par_iter().for_each(|run| {
        let mut attempt = 0;
        let mut found_perfect = false;

        while !found_perfect && attempt <= max_attempts {
            attempt += 1;
            if attempt > 1 {
                println!("Run {} - Tentativa {} (solução não perfeita, tentando novamente)", run, attempt);
            }

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
                println!("Run {} (Tentativa {}) encontrou solução perfeita na Geração {}!", run, attempt, r#gen);
                found_perfect = true;
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

            // Use tournament selection for better convergence
            let selecionados = tournament_selection(&evaluated, evaluated.len() / 2, 3);

            let mut offspring = apply_crossover_perm(&selecionados, crossover_prob);

            mutation_perm(&mut offspring, mutation_prob);

            // Elitism: keep the best individuals from the current generation
            let elitism_count = (pop as f64 * 0.1).max(2.0) as usize; // Keep top 10% or at least 2
            let mut sorted_evaluated = evaluated.clone();
            sorted_evaluated.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            
            // Combine elite individuals with offspring
            let mut next_gen: Vec<IndivPerm> = sorted_evaluated.into_iter().take(elitism_count).collect();
            next_gen.extend(offspring.into_iter().take(pop - elitism_count));

            population = next_gen
                .into_iter()
                .map(|ind| Representation::IntPerm(ind.genes))
                .collect();
            }

            if let Some(best_genes) = &global_best_genes {
                let attacks = perfect_fitness - global_best_score;
                println!(
                    "Run {} (Tentativa {}) -> Melhor fitness = {:.0} (Perfeito é {:.0}) | Ataques = {:.0}",
                    run,
                    attempt,
                    global_best_score,
                    perfect_fitness,
                    attacks
                );
                
                // Only save results if perfect solution is found
                if attacks == 0.0 {
                    // Create results directory if it doesn't exist
                    std::fs::create_dir_all("results").ok();
                    
                    let mut w = writer.lock().unwrap();
                    for (generation, best) in best_per_gen.iter().enumerate() {
                        w.write_record(&[
                            run.to_string(),
                            (generation + 1).to_string(), 
                            best.to_string(),
                            n_queens.to_string(),
                            crossover_prob.to_string(),
                            mutation_prob.to_string(),
                        ]).unwrap();
                    }
                    drop(w);
                    
                    let filename = format!("results/convergencia_nqueens_n{}_cx{}_mut{}_run{}_attempt{}.png", n_queens, crossover_prob, mutation_prob, run, attempt);
                    let plot_best: Vec<f64> = best_per_gen.iter().map(|&f| f / perfect_fitness).collect();
                    let plot_mean: Vec<f64> = mean_per_gen.iter().map(|&f| f / perfect_fitness).collect();
                    let plot_worst: Vec<f64> = worst_per_gen.iter().map(|&f| f / perfect_fitness).collect();
                    
                    plot_convergence(&plot_best, &plot_mean, &plot_worst, &filename);
                    
                    let genes_0_indexed: Vec<i32> = best_genes.iter().map(|&g| g - 1).collect();
                }
            }

            // Check if we found perfect solution or reached max attempts
            if found_perfect || attempt > max_attempts {
                break;
            }
        }
    });
}
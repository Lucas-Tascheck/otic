use crate::utils::functions::*;
use rayon::prelude::*;
use std::sync::Mutex;
use csv::Writer;
use rand::prelude::*;
use rand::Rng;
use std::collections::HashSet;

#[derive(Clone, Debug)]
struct IndivPermValued {
    pub genes: Vec<i32>,
    pub attacks: f64,
    pub board_value: f64,
    pub fitness: f64, // Combined fitness
}

/// Create a valued board where:
/// - Odd rows (1,3,5,7...): apply sqrt to column values
/// - Even rows (2,4,6,8...): apply log10 to column values
fn create_valued_board(n: usize) -> Vec<Vec<f64>> {
    let mut board = vec![vec![0.0; n]; n];
    
    for row in 0..n {
        for col in 0..n {
            let base_value = (col + 1) as f64; // Column position 1-indexed
            
            if row % 2 == 0 {
                // Odd row (0-indexed row 0 = 1st row): sqrt
                board[row][col] = base_value.sqrt();
            } else {
                // Even row (0-indexed row 1 = 2nd row): log10
                board[row][col] = base_value.log10();
            }
        }
    }
    
    board
}

/// Calculate the number of diagonal attacks
fn count_attacks(genes: &Vec<i32>) -> f64 {
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

    attacks as f64
}

/// Calculate the sum of board values where queens are placed
fn calculate_board_value(genes: &Vec<i32>, board: &Vec<Vec<f64>>) -> f64 {
    let mut total = 0.0;
    
    for (col, &row) in genes.iter().enumerate() {
        let row_idx = row as usize;
        total += board[row_idx][col];
    }
    
    total
}

/// Calculate the maximum possible board value (sum of max values in each column)
fn calculate_max_possible_value(board: &Vec<Vec<f64>>) -> f64 {
    let n = board.len();
    let mut max_value = 0.0;
    
    for col in 0..n {
        let mut col_max = f64::NEG_INFINITY;
        for row in 0..n {
            col_max = col_max.max(board[row][col]);
        }
        max_value += col_max;
    }
    
    max_value
}

/// Evaluate individual: minimize attacks, maximize board value
fn evaluate_valued_nqueens(genes: &Vec<i32>, board: &Vec<Vec<f64>>) -> IndivPermValued {
    let attacks = count_attacks(genes);
    let board_value = calculate_board_value(genes, board);
    
    // Fitness function: prioritize zero attacks, then maximize board value
    // If attacks > 0, fitness is negative (penalized)
    // If attacks == 0, fitness is the board value
    let fitness = if attacks > 0.0 {
        -attacks * 1000.0 - board_value // Heavy penalty for attacks
    } else {
        board_value // Maximize board value when no attacks
    };
    
    IndivPermValued {
        genes: genes.clone(),
        attacks,
        board_value,
        fitness,
    }
}

fn tournament_selection_valued(population: &[IndivPermValued], num_pairs: usize, tournament_size: usize) -> Vec<(&IndivPermValued, &IndivPermValued)> {
    let mut rng = rand::thread_rng();
    let mut selected = Vec::with_capacity(num_pairs);

    for _ in 0..num_pairs {
        let parent1 = select_one_by_tournament_valued(population, tournament_size, &mut rng);
        let parent2 = select_one_by_tournament_valued(population, tournament_size, &mut rng);
        selected.push((parent1, parent2));
    }
    selected
}

fn select_one_by_tournament_valued<'a>(
    population: &'a [IndivPermValued],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> &'a IndivPermValued {
    let mut best: Option<&IndivPermValued> = None;
    for _ in 0..tournament_size {
        let candidate = &population[rng.gen_range(0..population.len())];
        if best.is_none() || candidate.fitness > best.unwrap().fitness {
            best = Some(candidate);
        }
    }
    best.unwrap()
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

fn apply_crossover_perm_valued<'a>(pairs: &[(&'a IndivPermValued, &'a IndivPermValued)], crossover_prob: f64) -> Vec<Vec<i32>> {
    let mut rng = rand::thread_rng();
    let mut offspring = Vec::with_capacity(pairs.len() * 2);

    for (p1, p2) in pairs {
        if rng.gen_bool(crossover_prob) {
            let (genes1, genes2) = ox1_crossover(&p1.genes, &p2.genes);
            offspring.push(genes1);
            offspring.push(genes2);
        } else {
            offspring.push(p1.genes.clone());
            offspring.push(p2.genes.clone());
        }
    }
    offspring
}

fn mutation_perm_valued(population: &mut [Vec<i32>], mutation_prob: f64) {
    let mut rng = rand::thread_rng();
    let n = population[0].len();
    if n == 0 { return; }

    for genes in population.iter_mut() {
        if rng.gen_bool(mutation_prob) {
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            if i != j {
                genes.swap(i, j);
            }
        }
    }
}

fn print_board_valued(genes: &Vec<i32>, board: &Vec<Vec<f64>>) {
    let n = genes.len();
    println!("--- Solução N-Rainhas Valoradas (N={}) ---", n);
    
    for row in 0..n {
        let mut line = String::new();
        for col in 0..n {
            if genes[col] as usize == row {
                line.push_str(&format!(" Q({:.2}) ", board[row][col]));
            } else {
                line.push_str(&format!(" .({:.2}) ", board[row][col]));
            }
        }
        println!("{}", line);
    }
    
    let total_value = calculate_board_value(genes, board);
    println!("Valor Total do Tabuleiro: {:.4}", total_value);
    println!("---------------------------------");
}

pub fn run_nqueens_valued(n_queens: usize, pop: usize, gens: usize, runs: usize, crossover_prob: f64, mutation_prob: f64, max_attempts: usize) {
    let board = create_valued_board(n_queens);
    let max_possible_value = calculate_max_possible_value(&board);
    
    println!("=== Problema N-Rainhas Valoradas (N={}) ===", n_queens);
    println!("Valor Máximo Possível: {:.4}", max_possible_value);
    
    let csv_filename = format!("boxplot_nqueens_valued_n{}_cx{}_mut{}.csv", n_queens, crossover_prob, mutation_prob);
    let writer = Mutex::new({
        let mut wtr = Writer::from_path(&csv_filename).unwrap();
        wtr.write_record(&["Run", "Generation", "Best_Fitness", "Attacks", "Board_Value", &format!("N={}", n_queens), &format!("Crossover={}", crossover_prob), &format!("Mutation={}", mutation_prob)]).unwrap();
        wtr
    });

    (1..=runs).into_par_iter().for_each(|run| {
        let mut attempt = 0;
        let mut found_perfect = false;

        while !found_perfect && attempt <= max_attempts {
            attempt += 1;

            let mut global_best: Option<IndivPermValued> = None;

            let mut population = generate_population(pop, RepresentationType::IntPerm { dim: n_queens });

            let mut best_per_gen = Vec::with_capacity(gens);
            let mut mean_per_gen = Vec::with_capacity(gens);
            let mut worst_per_gen = Vec::with_capacity(gens);

            for r#gen in 1..=gens {
                let evaluated: Vec<IndivPermValued> = population.par_iter().map(|ind| {
                    if let Representation::IntPerm(genes) = ind {
                        let genes_0_indexed: Vec<i32> = genes.iter().map(|&g| g - 1).collect();
                        evaluate_valued_nqueens(&genes_0_indexed, &board)
                    } else { panic!("Representação inválida, use IntPerm."); }
                }).collect();

                // Track global best
                for ind in &evaluated {
                    if global_best.is_none() || ind.fitness > global_best.as_ref().unwrap().fitness {
                        global_best = Some(ind.clone());
                    }
                }

                let best = evaluated.iter().map(|ind| ind.fitness).fold(f64::MIN, f64::max);
                let worst = evaluated.iter().map(|ind| ind.fitness).fold(f64::MAX, f64::min);
                let mean = evaluated.iter().map(|ind| ind.fitness).sum::<f64>() / evaluated.len() as f64;

                best_per_gen.push(best);
                mean_per_gen.push(mean);
                worst_per_gen.push(worst);

                // Check if we found a perfect solution (no attacks)
                if let Some(ref best_ind) = global_best {
                    if best_ind.attacks == 0.0 {
                        found_perfect = true;
                        
                        // Fill remaining generations with final values
                        for _ in r#gen..=gens {
                            best_per_gen.push(best);
                            mean_per_gen.push(mean);
                            worst_per_gen.push(worst);
                        }
                        break;
                    }
                }

                // Tournament selection
                let selecionados = tournament_selection_valued(&evaluated, evaluated.len() / 2, 3);

                let mut offspring_genes = apply_crossover_perm_valued(&selecionados, crossover_prob);

                mutation_perm_valued(&mut offspring_genes, mutation_prob);

                // Re-evaluate offspring
                let mut offspring: Vec<IndivPermValued> = offspring_genes.par_iter().map(|genes| {
                    evaluate_valued_nqueens(genes, &board)
                }).collect();

                // Elitism: keep top 10% from current generation
                let elitism_count = (pop as f64 * 0.1).max(2.0) as usize;
                let mut sorted_evaluated = evaluated.clone();
                sorted_evaluated.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
                
                let mut next_gen: Vec<IndivPermValued> = sorted_evaluated.into_iter().take(elitism_count).collect();
                next_gen.extend(offspring.into_iter().take(pop - elitism_count));

                population = next_gen
                    .into_iter()
                    .map(|ind| {
                        let genes_1_indexed: Vec<i32> = ind.genes.iter().map(|&g| g + 1).collect();
                        Representation::IntPerm(genes_1_indexed)
                    })
                    .collect();
            }

            if let Some(best_ind) = &global_best {
                // Only save results if perfect solution is found (no attacks)
                if best_ind.attacks == 0.0 {
                    let percentage = (best_ind.board_value / max_possible_value) * 100.0;
                    println!("Run {} (Tentativa {}) -> Fitness = {:.4} | Ataques = {:.0} | Valor = {:.4} ({:.2}% do máximo)",
                        run, attempt, best_ind.fitness, best_ind.attacks, best_ind.board_value, percentage);
                    
                    std::fs::create_dir_all("results").ok();
                    
                    let mut w = writer.lock().unwrap();
                    for (generation, &best_fit) in best_per_gen.iter().enumerate() {
                        w.write_record(&[
                            run.to_string(),
                            (generation + 1).to_string(), 
                            best_fit.to_string(),
                            best_ind.attacks.to_string(),
                            best_ind.board_value.to_string(),
                            n_queens.to_string(),
                            crossover_prob.to_string(),
                            mutation_prob.to_string(),
                        ]).unwrap();
                    }
                    drop(w);
                    
                    let filename = format!("results/convergencia_nqueens_valued_n{}_cx{}_mut{}_run{}_attempt{}.png", 
                        n_queens, crossover_prob, mutation_prob, run, attempt);
                    
                    // Normalize fitness for plotting
                    let max_fitness = best_per_gen.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let min_fitness = worst_per_gen.iter().cloned().fold(f64::INFINITY, f64::min);
                    let range = (max_fitness - min_fitness).max(1.0);
                    
                    let plot_best: Vec<f64> = best_per_gen.iter().map(|&f| (f - min_fitness) / range).collect();
                    let plot_mean: Vec<f64> = mean_per_gen.iter().map(|&f| (f - min_fitness) / range).collect();
                    let plot_worst: Vec<f64> = worst_per_gen.iter().map(|&f| (f - min_fitness) / range).collect();
                    
                    plot_convergence(&plot_best, &plot_mean, &plot_worst, &filename);
                }
            }

            if found_perfect || attempt > max_attempts {
                break;
            }
        }
    });
}

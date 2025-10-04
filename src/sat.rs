use crate::utils::functions::*;
use rayon::prelude::*;
use bitvec::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone)]
pub struct Clause {
    pub vars: [i32; 3],
}

#[derive(Debug, Clone)]
pub struct SATInstance {
    pub clauses: Vec<Clause>,
    pub n_vars: usize,
}

// --- Carrega CNF ---
pub fn load_cnf(path: &str) -> SATInstance {
    let file = File::open(path).expect("Erro ao abrir arquivo CNF");
    let reader = BufReader::new(file);

    let mut n_vars = 0;
    let mut clauses: Vec<Clause> = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let line = line.trim();

        if line.is_empty() || line.starts_with('c') {
            continue;
        }

        if line.starts_with('p') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            n_vars = parts[2].parse::<usize>().unwrap();
        } else {
            let nums: Vec<i32> = line
                .split_whitespace()
                .filter_map(|x| x.parse::<i32>().ok())
                .take_while(|&x| x != 0)
                .collect();

            if nums.len() == 3 {
                clauses.push(Clause { vars: [nums[0], nums[1], nums[2]] });
            } else if !nums.is_empty() {
                panic!("Cláusula não é 3-SAT: {:?}", nums);
            }
        }
    }

    SATInstance { clauses, n_vars }
}

// --- Avalia um indivíduo ---
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

pub fn evaluate_individual(ind: &Representation, instance: &SATInstance) -> f64 {
    match ind {
        Representation::Binary(genes) => {
            let score = evaluate(instance, genes) as f64;
            let max_score = instance.clauses.len() as f64;
            score / max_score
        }
        _ => panic!("Representação inválida, use Binary."),
    }
}

// --- Executa GA para 3-SAT ---
pub fn run_3sat(pop: usize, gens: usize, runs: usize, crossover_prob: f64, mutation_prob: f64, cnf_path: &str) {
    println!("\n=== EX 3: Problema 3-SAT ===");

    (1..=runs).into_par_iter().for_each(|run| {
        let instance = load_cnf(cnf_path);
        let mut global_best_score = -1.0;
        let mut global_best_genes: Option<BitVec> = None;
        
        let mut population: Vec<Representation> = generate_population(
            pop,
            RepresentationType::Binary { dim: instance.n_vars },
        );

        let mut best_per_gen = Vec::with_capacity(gens);
        let mut mean_per_gen = Vec::with_capacity(gens);
        let mut worst_per_gen = Vec::with_capacity(gens);

        for _ in 0..=gens {
            let evaluated: Vec<Indiv> = population.par_iter().map(|ind| {
                let fitness = match ind {
                    Representation::Binary(genes) => evaluate_individual(&Representation::Binary(genes.clone()), &instance),
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

            //let selecionados = roulette(&evaluated, evaluated.len() / 2);
            let selecionados = tournament_selection(&evaluated, evaluated.len() / 2, 5);
            //let mut crossover = apply_crossover(&selecionados, crossover_prob);
            let mut crossover = uniform_crossover(&selecionados, 0.8);

            mutation(&mut crossover, mutation_prob);

            population = crossover
                .into_iter()
                .map(|ind| Representation::Binary(ind.genes))
                .collect();
        }

        let filename = format!("convergencia_3SAT_run{}.png", run);
        plot_convergence(&best_per_gen, &mean_per_gen, &worst_per_gen, &filename);

        if let Some(best_genes) = global_best_genes {
            let satisfied = evaluate(&instance, &best_genes);
            // let assignment: Vec<_> = best_genes.iter().map(|b| if *b { 1 } else { 0 }).collect();
            println!(
                "Run {} -> Melhor indivíduo: {:?} | Cláusulas satisfeitas = {}/{} | fitness = {:.4}",
                run,
                print_bits(&best_genes),
                satisfied,
                instance.clauses.len(),
                global_best_score
            );
        }
    });
}

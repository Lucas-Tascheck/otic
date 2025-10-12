use crate::utils::functions::*;
use rayon::prelude::*;
use rand::Rng;
use bitvec::prelude::BitVec;



pub fn zdt1_discreta(x: &[i32]) -> (f64, f64) {
    let n = x.len();
    assert!(n >= 1, "vetor deve ter pelo menos 1 variável");
    
    let f1 = x[0] as f64 / 1000.0;
    
    let g = if n == 1 {
        1.0
    } else {
        let sum_tail: f64 = x.iter().skip(1).map(|&xi| xi as f64 / 1000.0).sum();
        1.0 + 9.0 * sum_tail / ((n - 1) as f64)
    };
    
    let ratio = (f1 / g);
    let f2 = g * (1.0 - ratio.sqrt());
    
    (f1, f2)
}

pub fn evaluate_individual(genes: &BitVec, bit_length: u32) -> Vec<f64> {
    let n_vars = (genes.len() as u32) / bit_length;
    let mut x_vals: Vec<i32> = Vec::with_capacity(n_vars as usize);

    for i in 0..n_vars {
        let start = (i * bit_length) as usize;
        let end = ((i + 1) * bit_length) as usize;
        let slice = &genes[start..end];
        let val = gen_to_fen(slice, 0, 1000, bit_length).round() as i32;
        x_vals.push(val);
    }

    let (f1, f2) = zdt1_discreta(&x_vals);
    vec![f1, f2]
}

fn pareto_front(pop: &[IndivMO]) -> Vec<IndivMO> {
    let mut front = Vec::new();
    for ind in pop {
        if !pop.iter().any(|other| dominates(other, ind) && other != ind) {
            front.push(ind.clone());
        }
    }
    front
}

pub fn run_multi_objetiva(
    pop: usize,
    dim: usize,
    gens: usize,
    runs: usize,
    crossover_prob: f64,
    mutation_prob: f64,
) {
    println!("=== EX: ZDT1 Multiobjetivo ===");

    (1..=runs).into_par_iter().for_each(|run| {
        // Gera população inicial
        let mut population = generate_population(pop, RepresentationType::Binary { dim });

        for g in 1..=gens {
            // === Avaliação ===
            let evaluated: Vec<IndivMO> = population
                .par_iter()
                .map(|ind| {
                    if let Representation::Binary(genes) = ind {
                        let objectives = evaluate_individual(genes, 10);
                        IndivMO::new(genes.clone(), objectives)
                    } else {
                        unreachable!()
                    }
                })
                .collect();

            let filename = format!("generation_{:03}.png", g);
            plot_population_colored(&evaluated, g, &filename);

            // === Fronte de Pareto ===
            let pareto = pareto_front(&evaluated);
            println!("Geração {} -> Fronte de Pareto ({} indivíduos):", g, pareto.len());
            for ind in &pareto {
                println!(" f1 = {:.4}, f2 = {:.4}", ind.objectives[0], ind.objectives[1]);
            }

            // === Seleção, crossover e mutação ===
            // Mantém o tamanho da população constante
            let selecionados = tournament_selection_mo(&evaluated, pop, 3);
            let mut offspring = apply_crossover_mo(&selecionados, crossover_prob);
            mutation_mo(&mut offspring, mutation_prob);

            // Se por algum motivo o crossover gerou menos indivíduos, completa com os selecionados
            if offspring.len() < pop {
                let deficit = pop - offspring.len();
                offspring.extend(selecionados.into_iter().take(deficit));
            } else if offspring.len() > pop {
                offspring.truncate(pop);
            }

            // Atualiza população
            population = offspring.into_iter().map(|ind| Representation::Binary(ind.genes)).collect();
        }

        println!("=== Run {} finalizado ===", run);
    });
}

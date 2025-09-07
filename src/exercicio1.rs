use crate::initialize_population::{Representation, RepresentationType, generate_population, Config, read_config};

pub fn evaluate_individual(ind: &Representation) -> f64 {
    let f_min = -4.0;
    let f_max = 2.0;

    match ind {
        Representation::Real(genes) => {
            let mut total = 0.0;
            for &x in genes {
                let fx = (20.0 * x).cos() - x.abs() / 2.0 + (x.powi(3) / 4.0);

                let fx_norm = (fx - f_min) / (f_max - f_min);
                total += fx_norm;
            }
            total / genes.len() as f64
        }
        _ => panic!("Representação inválida para este exercício, use Real."),
    }
}

pub fn run_exercicio1(pop: usize, dim: usize) {
    println!("=== EX 1: Maximização de Função Algébrica ===");

    let population = generate_population(
        pop,
        RepresentationType::Real { dim, min: -2.0, max: 2.0 }
    );

    for (i, ind) in population.iter().enumerate() {
        let score = evaluate_individual(ind);
        println!("Indivíduo {}: {:?} -> score normalizado = {:.4}", i + 1, ind, score);
    }
}

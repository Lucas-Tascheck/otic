mod initialize_population;
mod sat;

use initialize_population::*;
use sat::*;

fn main() {
    let data = read_config("./src/config.json");

    let pop_size = data.pop;
    let dim = data.dim;

    println!("=== EX 1: Populações iniciais ===");
    let pop_bin = generate_population(10, RepresentationType::Binary { dim: 15 });
    let pop_int = generate_population(10, RepresentationType::Integer { dim: 15, min: -5, max: 10 });
    let pop_perm = generate_population(10, RepresentationType::IntPerm { dim: 15 });
    let pop_real = generate_population(10, RepresentationType::Real { dim: 15, min: -10.0, max: 10.0 });

    println!("Exemplo BIN: {:?}", pop_bin[0]);
    println!("Exemplo INT: {:?}", pop_int[0]);
    println!("Exemplo PERM: {:?}", pop_perm[0]);
    println!("Exemplo REAL: {:?}", pop_real[0]);

    println!("\n=== EX 2: Problema 3-SAT ===");
    let instance = toy_instance();

    let pop = generate_population(pop_size, RepresentationType::Binary { dim: instance.n_vars });

    let mut best_score = -1;
    let mut best_ind: Vec<u8> = Vec::new();

    for ind in &pop {
        if let Representation::Binary(genes) = ind {
            let score = evaluate(&instance, genes);
            if score > best_score {
                best_score = score;
                best_ind = genes.clone();
            }
        }
    }

    println!(
        "Melhor indivíduo: {:?} com {} cláusulas satisfeitas",
        best_ind, best_score
    );
}

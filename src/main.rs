mod initialize_population;
mod sat;

use initialize_population::*;
use sat::*;

fn main() {
    println!("=== EX 1: Populações iniciais ===");
    let pop_bin = generate_binary(10, 15);
    let pop_int = generate_integer(10, 15, -5, 10);
    let pop_perm = generate_intperm(10, 15);
    let pop_real = generate_real(10, 15, -10.0, 10.0);

    println!("Exemplo BIN: {:?}", pop_bin[0]);
    println!("Exemplo INT: {:?}", pop_int[0]);
    println!("Exemplo PERM: {:?}", pop_perm[0]);
    println!("Exemplo REAL: {:?}", pop_real[0]);

    println!("\n=== EX 2: Problema 3-SAT ===");
    let instance = toy_instance();
    let pop = generate_population(30, instance.n_vars);

    let mut best_score = -1;
    let mut best_ind: Vec<u8> = Vec::new();

    for ind in pop {
        let score = evaluate(&instance, &ind);
        if score > best_score {
            best_score = score;
            best_ind = ind.clone();
        }
    }

    println!(
        "Melhor indivíduo: {:?} com {} cláusulas satisfeitas",
        best_ind, best_score
    );
}

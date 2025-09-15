mod initialize_population;
mod exercicio1;
mod sat;

use initialize_population::*;
use sat::*;
use exercicio1::*;

fn main() {
    let data = read_config("./src/config.json");

    let pop_size = data.pop;
    let dim = data.dim;
    let runs = data.runs;
    let gens = data.gens;

    println!("=== EX 1: Populações iniciais ===");
    let pop_bin = generate_population(10, RepresentationType::Binary { dim: 15 });
    let pop_int = generate_population(10, RepresentationType::Integer { dim: 15, min: -5, max: 10 });
    let pop_perm = generate_population(10, RepresentationType::IntPerm { dim: 15 });
    let pop_real = generate_population(10, RepresentationType::Real { dim: 15, min: -10.0, max: 10.0 });

    println!("Exemplo BIN: {:?}", pop_bin[0]);
    println!("Exemplo INT: {:?}", pop_int[0]);
    println!("Exemplo PERM: {:?}", pop_perm[0]);
    println!("Exemplo REAL: {:?}", pop_real[0]);

    // Parametros: POP, DIM, GENS, RUNS

    // Exercicio 3 SAT
    //run_3sat(30, 3, gens, runs);

    // EX 1: Maximização de Função Algébrica 
    // valor otimo para X = 1.8904170901022
    run_exercicio1(100, 1, gens, runs);
}

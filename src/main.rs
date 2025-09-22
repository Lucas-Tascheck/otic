mod functions;
mod exercicio1;
mod exercicio2;
mod sat;

use functions::*;
use sat::*;
use exercicio1::*;
use exercicio2::*;

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
    // minimize: c_max = 2.0; 
    //           ((c_max - fx)/normalize).max(0.0)
    // maximize:
    //           c_min = 4.0;
    //           ((fx + c_min)/normalize).max(0.0)
    run_exercicio1(pop_size, dim, gens, runs);

    //EX 2: Fabrica de Rádios
    //Maximização: genes = "1111111110" | fitness = 0.7571 | Lucro = 1029.68
    // 11111 = 24 st
    // 11110 = ~8 lx
    // Lucro = 1040.0
    // Objective function = (30.0 * st + 40.0 * lx) / 1360.0
    // Penalty funciton = ((st + 2.0 * lx - 40.0) / 16.0).max(0.0)
    // Fitness = Objective function - k * Penalty funciton
    //run_exercicio2(pop_size, dim, gens, runs, 1.0)
}

mod utils;
mod exercicios;
mod sat;

use utils::functions::*;
use sat::*;
use exercicios::exercicio1::*;
use exercicios::exercicio2::*;
use exercicios::multi_objetiva::*;
use exercicios::nqueens::*;
use exercicios::nqueens_valued::*;

fn main() {
    let data = read_config("./src/config.json");

    let pop_size = data.pop;
    let dim = data.dim;
    let runs = data.runs;
    let gens = data.gens;

    // println!("=== EX 1: Populações iniciais ===");
    // let pop_bin = generate_population(10, RepresentationType::Binary { dim: 15 });
    // let pop_int = generate_population(10, RepresentationType::Integer { dim: 15, min: -5, max: 10 });
    // let pop_perm = generate_population(10, RepresentationType::IntPerm { dim: 15 });
    // let pop_real = generate_population(10, RepresentationType::Real { dim: 15, min: -10.0, max: 10.0 });

    // println!("Exemplo BIN: {:?}", pop_bin[0]);
    // println!("Exemplo INT: {:?}", pop_int[0]);
    // println!("Exemplo PERM: {:?}", pop_perm[0]);
    // println!("Exemplo REAL: {:?}", pop_real[0]);

    // Parametros: POP, DIM, GENS, RUNS

    // Exercicio 3 SAT
    //run_3sat(pop_size, gens, runs, 0.8, 0.015, "./sat.cnf");

    // EX 1: Maximização de Função Algébrica 
    // valor otimo para X = 1.8904170901022
    // minimize: c_max = 2.0; 
    //           ((c_max - fx)/normalize).max(0.0)
    // maximize:
    //           c_min = 4.0;
    //           ((fx + c_min)/normalize).max(0.0)
    //run_exercicio1(pop_size, 16, gens, runs, 0.8, 0.005, true);

    //EX 2: Fabrica de Rádios
    //Maximização: genes = "1111111110" | fitness = 0.7571 | Lucro = 1029.68
    // 11111 = 24 st
    // 11110 = ~8 lx
    // Lucro = 1040.0
    // Objective function = (30.0 * st + 40.0 * lx) / 1360.0
    // Penalty funciton = ((st + 2.0 * lx - 40.0) / 16.0).max(0.0)
    // Fitness = Objective function - k * Penalty funciton
    // run_exercicio2(pop_size, 10, gens, runs, 0.9, 0.004, 1.0);

    //EX Multi Objetivo:
    //( População, Num. de valores (Xi), tamanho do Bit, gerações, runs, Prob. Crossover, Prob. Mutação )
    // run_zdt1(pop_size, 30, 10, gens, runs, 0.9, 0.01);
    // run_zdt3(pop_size, 30, 10, gens, runs, 0.9, 0.01);

    //EX N-Queens
    //Fitness: minimizar os pares sob ataque
    //Numero de pares unicos N * (N-1) / 2, onde N eh o numero de rainhas
    //Fitness = (Pontuação Perfeita) - (Nº de Ataques Diagonais)
    //Order 1 Crossover (OX1).
    //Swap Mutation (Mutação por Troca).
    // With elitism and tournament selection, mutation is now per-individual
    // run_nqueens(128, pop_size, gens, runs, 0.9, 0.08, 10);

    //EX N-Queens Valued (Multi-objective)
    //Minimize attacks AND maximize board position values
    //Board values: odd rows use sqrt, even rows use log10
    //Parameters: N-Queens, Population, Generations, Runs, Crossover Prob, Mutation Prob, Max Attempts
    
    // Test with N=8 and N=16 as specified
    run_nqueens_valued(8, 30, gens, 10, 0.9, 0.10, 5);
    // run_nqueens_valued(16, 30, gens, 10, 0.9, 0.15, 3);
}

//TODO:
// Otimizar N-Queens (32x32 ja quebra legal)
// Procurar se tem metodos de crossover e mutacao melhores
// Ver se tem outra opcao mais eficiente de calculo da fitness
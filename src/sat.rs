use rand::Rng;

#[derive(Debug, Clone)]
pub struct Clause {
    pub vars: [i32; 3], // positivo = normal, negativo = negado
}

#[derive(Debug)]
pub struct SATInstance {
    pub clauses: Vec<Clause>,
    pub n_vars: usize,
}

pub fn toy_instance() -> SATInstance {
    SATInstance {
        n_vars: 4,
        clauses: vec![
            Clause { vars: [1, -2, 3] },   // (x1 ∨ ¬x2 ∨ x3)
            Clause { vars: [-1, 2, 4] },   // (¬x1 ∨ x2 ∨ x4)
            Clause { vars: [-3, -4, 2] },  // (¬x3 ∨ ¬x4 ∨ x2)
        ],
    }
}

pub fn evaluate(instance: &SATInstance, assignment: &Vec<u8>) -> i32 {
    let mut satisfied = 0;
    for clause in &instance.clauses {
        let mut clause_sat = false;
        for &var in &clause.vars {
            let idx = var.abs() as usize - 1;
            let val = assignment[idx] == 1;
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

pub fn generate_population(pop_size: usize, n_vars: usize) -> Vec<Vec<u8>> {
    let mut rng = rand::thread_rng();
    (0..pop_size)
        .map(|_| (0..n_vars).map(|_| rng.gen_range(0..=1)).collect())
        .collect()
}

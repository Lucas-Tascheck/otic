// src/main.rs
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use bitvec::prelude::*;
use plotters::prelude::*;
use std::collections::HashMap;
use crate::utils::functions::*;
use plotters::style::{TextStyle, IntoFont};
use std::fs::File;
use std::io::Write;

fn dominates(a: &IndivMO, b: &IndivMO) -> bool {
    let mut strictly_better = false;
    for i in 0..a.objectives.len() {
        if a.objectives[i] > b.objectives[i] {
            return false;
        }
        if a.objectives[i] < b.objectives[i] {
            strictly_better = true;
        }
    }
    strictly_better
}

fn non_dominated_sort(pop: &[IndivMO]) -> Vec<Vec<usize>> {
    let n = pop.len();
    let mut domination_count = vec![0usize; n];
    let mut dominated_sets: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    
    for p in 0..n {
        for q in 0..n {
            if p == q { continue; }
            if dominates(&pop[p], &pop[q]) {
                dominated_sets[p].push(q);
            } else if dominates(&pop[q], &pop[p]) {
                domination_count[p] += 1;
            }
        }
        if domination_count[p] == 0 {
            if fronts.is_empty() {
                fronts.push(Vec::new());
            }
            fronts[0].push(p);
        }
    }
    
    let mut i = 0;
    while i < fronts.len() {
        let mut next_front: Vec<usize> = Vec::new();
        for &p in &fronts[i] {
            for &q in &dominated_sets[p] {
                domination_count[q] = domination_count[q].saturating_sub(1);
                if domination_count[q] == 0 {
                    next_front.push(q);
                }
            }
        }
        if !next_front.is_empty() {
            fronts.push(next_front);
        }
        i += 1;
    }
    
    fronts
}

fn crowding_distance_for_front(pop: &[IndivMO], front: &[usize]) -> HashMap<usize, f64> {
    let mut distances: HashMap<usize, f64> = HashMap::new();
    let m = pop[0].objectives.len();
    
    if front.is_empty() {
        return distances;
    }
    if front.len() == 1 {
        distances.insert(front[0], f64::INFINITY);
        return distances;
    }

    for &idx in front {
        distances.insert(idx, 0.0);
    }
    
    for obj in 0..m {
        let mut front_sorted = front.to_vec();
        front_sorted.sort_by(|&a, &b| {
            pop[a].objectives[obj]
            .partial_cmp(&pop[b].objectives[obj])
            .unwrap()
        });

        let f_min = pop[*front_sorted.first().unwrap()].objectives[obj];
        let f_max = pop[*front_sorted.last().unwrap()].objectives[obj];
        
        distances.insert(front_sorted[0], f64::INFINITY);
        distances.insert(front_sorted[front_sorted.len() - 1], f64::INFINITY);
        
        for j in 1..(front_sorted.len() - 1) {
            let prev = pop[front_sorted[j - 1]].objectives[obj];
            let next = pop[front_sorted[j + 1]].objectives[obj];
            let denom = if (f_max - f_min).abs() < 1e-12 { 1.0 } else { f_max - f_min };
            let add = (next - prev) / denom;
            let cur = distances.get(&front_sorted[j]).copied().unwrap_or(0.0);
            distances.insert(front_sorted[j], cur + add);
        }
    }
    
    distances
}

fn rank_and_crowding(pop: &[IndivMO]) -> (Vec<usize>, Vec<f64>) {
    let fronts = non_dominated_sort(pop);
    let mut rank = vec![usize::MAX; pop.len()];
    let mut crowd = vec![0.0f64; pop.len()];
    
    for (i, front) in fronts.iter().enumerate() {
        for &idx in front {
            rank[idx] = i;
        }
        let dmap = crowding_distance_for_front(pop, front);
        for (&idx, &d) in &dmap {
            crowd[idx] = if d.is_infinite() { f64::INFINITY } else { d };
        }
    }
    (rank, crowd)
}

fn tournament_binary(pop: &[IndivMO], rank: &[usize], crowd: &[f64]) -> usize {
    let mut rng = rand::thread_rng();
    let a = rng.gen_range(0..pop.len());
    let b = rng.gen_range(0..pop.len());
    if rank[a] < rank[b] {
        a
    } else if rank[b] < rank[a] {
        b
    } else {
        if crowd[a] > crowd[b] { a } else { b }
    }
}

fn single_point_crossover(parent1: &BitVec, parent2: &BitVec) -> (BitVec, BitVec) {
    let len = parent1.len();
    let mut rng = rand::thread_rng();
    if len < 2 {
        return (parent1.clone(), parent2.clone());
    }
    let point = rng.gen_range(1..len);
    let mut child1 = parent1.clone();
    let mut child2 = parent2.clone();
    for i in point..len {
        let tmp = child1[i];
        child1.set(i, child2[i]);
        child2.set(i, tmp);
    }
    (child1, child2)
}

fn mutate_bitvec(bits: &mut BitVec, mutation_prob: f64) {
    let mut rng = rand::thread_rng();
    for i in 0..bits.len() {
        if rng.gen_bool(mutation_prob) {
            let cur = bits[i];
            bits.set(i, !cur);
        }
    }
}

fn make_offspring(pop: &[IndivMO], pop_size: usize, crossover_prob: f64, mutation_prob: f64, bit_length: u32, evaluate: fn(&BitVec, u32) -> Vec<f64>) -> Vec<IndivMO> {
    let (rank, crowd) = rank_and_crowding(pop);
    let mut offspring: Vec<IndivMO> = Vec::with_capacity(pop_size);
    let mut rng = rand::thread_rng();
    
    while offspring.len() < pop_size {
        let p1_idx = tournament_binary(pop, &rank, &crowd);
        let p2_idx = tournament_binary(pop, &rank, &crowd);
        
        let parent1 = &pop[p1_idx].genes;
        let parent2 = &pop[p2_idx].genes;
        
        if rng.gen_bool(crossover_prob) {
            let (mut c1, mut c2) = single_point_crossover(parent1, parent2);
            mutate_bitvec(&mut c1, mutation_prob);
            mutate_bitvec(&mut c2, mutation_prob);
            
            let o1 = evaluate(&c1, bit_length);
            let o2 = evaluate(&c2, bit_length);
            offspring.push(IndivMO::new(c1, o1));
            if offspring.len() < pop_size {
                offspring.push(IndivMO::new(c2, o2));
            }
        } else {
            let mut c1 = parent1.clone();
            let mut c2 = parent2.clone();
            mutate_bitvec(&mut c1, mutation_prob);
            mutate_bitvec(&mut c2, mutation_prob);
            let o1 = evaluate(&c1, bit_length);
            let o2 = evaluate(&c2, bit_length);
            offspring.push(IndivMO::new(c1, o1));
            if offspring.len() < pop_size {
                offspring.push(IndivMO::new(c2, o2));
            }
        }
    }
    
    offspring.truncate(pop_size);
    offspring
}

fn nsga2_select(next_pop_size: usize, parents: &[IndivMO], children: &[IndivMO]) -> Vec<IndivMO> {
    let mut combined: Vec<IndivMO> = Vec::with_capacity(parents.len() + children.len());
    combined.extend_from_slice(parents);
    combined.extend_from_slice(children);
    
    let fronts = non_dominated_sort(&combined);
    let mut new_pop: Vec<IndivMO> = Vec::with_capacity(next_pop_size);
    
    for front in fronts {
        if new_pop.len() + front.len() <= next_pop_size {
            for idx in front {
                new_pop.push(combined[idx].clone());
            }
        } else {
            let distances = crowding_distance_for_front(&combined, &front);
            let mut keyed: Vec<(usize, f64)> = front.into_iter()
            .map(|idx| (idx, *distances.get(&idx).unwrap_or(&0.0)))
            .collect();
            keyed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (idx, _) in keyed {
                if new_pop.len() < next_pop_size {
                    new_pop.push(combined[idx].clone());
                } else {
                    break;
                }
            }
            break;
        }
    }
    
    new_pop
}

fn nsga2_select_no_crowding(next_pop_size: usize, parents: &[IndivMO], children: &[IndivMO]) -> Vec<IndivMO> {
    let mut combined: Vec<IndivMO> = Vec::with_capacity(parents.len() + children.len());
    combined.extend_from_slice(parents);
    combined.extend_from_slice(children);

    let fronts = non_dominated_sort(&combined);
    let mut new_pop: Vec<IndivMO> = Vec::with_capacity(next_pop_size);
    let mut rng = rand::thread_rng();

    for front in fronts {
        if new_pop.len() + front.len() <= next_pop_size {
            for idx in front {
                new_pop.push(combined[idx].clone());
            }
        } else {
            let remaining = next_pop_size - new_pop.len();
            let mut front_indices = front.clone();
            front_indices.shuffle(&mut rng);
            for &idx in front_indices.iter().take(remaining) {
                new_pop.push(combined[idx].clone());
            }
            break;
        }
    }

    new_pop
}

pub fn plot_pareto_levels(
    pop: &[IndivMO],
    filename: &str,
    gens: usize,
    pop_size: usize,
    n_vars: usize,
) {
    let fronts = non_dominated_sort(pop);

    let f1_vals: Vec<f64> = pop.iter().map(|ind| ind.objectives[0]).collect();
    let f2_vals: Vec<f64> = pop.iter().map(|ind| ind.objectives[1]).collect();

    let (min_f1, max_f1) = f1_vals.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min, max), &v| (min.min(v), max.max(v)),
    );
    let (min_f2, max_f2) = f2_vals.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min, max), &v| (min.min(v), max.max(v)),
    );

    let margin_x = (max_f1 - min_f1) * 0.05;
    let margin_y = (max_f2 - min_f2) * 0.05;

    let x_range = (min_f1 - margin_x)..(max_f1 + margin_x);
    let y_range = (min_f2 - margin_y)..(max_f2 + margin_y);

    let root_area = BitMapBackend::new(filename, (1200, 850)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let (top_area, rest_area) = root_area.split_vertically(50);
    let (subtitle_area, plot_area) = rest_area.split_vertically(50);

    let title_text = "Fronte de Pareto (NSGA-II) - Níveis de Dominância";
    top_area
        .titled(
            title_text,
            ("sans-serif", 34).into_font().style(FontStyle::Bold),
        )
        .unwrap();

    let subtitle_text = format!(
        "Geração: {}   |   População: {}   |   Variáveis: {}",
        gens, pop_size, n_vars
    );

    let (w, h) = subtitle_area.dim_in_pixel();
    let text_style = TextStyle::from(("sans-serif", 30).into_font()).color(&BLACK);

    // mede o texto aproximadamente pela largura do caractere médio
    let est_text_width = subtitle_text.len() as i32 * 10; 
    let text_x = (w as i32 / 2) - (est_text_width / 2) + 20;
    let text_y = (h as i32 / 2) - 5;

    subtitle_area
        .draw_text(&subtitle_text, &text_style, (text_x, text_y))
        .unwrap();

    let mut chart = ChartBuilder::on(&plot_area)
        .caption("", ("sans-serif", 0))
        .margin(40)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(x_range.clone(), y_range.clone())
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("f1")
        .y_desc("f2")
        .label_style(("sans-serif", 16))
        .axis_desc_style(("sans-serif", 18))
        .draw()
        .unwrap();

    let palette: Vec<RGBColor> = vec![
        RGBColor(220, 50, 47),
        RGBColor(38, 139, 210),
        RGBColor(42, 161, 152),
        RGBColor(155, 89, 182),
        RGBColor(255, 165, 0),
        RGBColor(255, 215, 0),
        RGBColor(128, 128, 128),
        RGBColor(0, 0, 0),
    ];

    for (i, front) in fronts.iter().enumerate() {
        let color = palette[i % palette.len()];
        let pts: Vec<(f64, f64)> = front
            .iter()
            .map(|&idx| {
                let ind = &pop[idx];
                (ind.objectives[0], ind.objectives[1])
            })
            .collect();

        chart
            .draw_series(pts.iter().map(|&(x, y)| Circle::new((x, y), 4, color.filled())))
            .unwrap()
            .label(format!("Fronte {}", i))
            .legend(move |(x, y)| Circle::new((x, y), 5, color.filled()));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 18))
        .draw()
        .unwrap();

    println!(
        "✅ Plot salvo em '{}'. Eixos ajustados para [{:.3}–{:.3}] x [{:.3}–{:.3}]",
        filename, min_f1, max_f1, min_f2, max_f2
    );
}

fn spacing(pop: &[IndivMO]) -> f64 {
    if pop.len() < 2 {
        return 0.0;
    }

    let mut distances = Vec::new();
    for i in 0..pop.len() {
        let mut d_min = f64::INFINITY;
        for j in 0..pop.len() {
            if i != j {
                let d = (0..pop[i].objectives.len())
                    .map(|k| (pop[i].objectives[k] - pop[j].objectives[k]).abs())
                    .sum::<f64>();
                if d < d_min {
                    d_min = d;
                }
            }
        }
        distances.push(d_min);
    }

    let mean_d = distances.iter().sum::<f64>() / distances.len() as f64;
    let sum_sq = distances.iter().map(|&d| (d - mean_d).powi(2)).sum::<f64>();
    (sum_sq / (distances.len() as f64 - 1.0)).sqrt()
}

fn hypervolume(pop: &[IndivMO], reference: (f64, f64)) -> f64 {
    let mut pts: Vec<(f64, f64)> = pop.iter()
        .map(|ind| (ind.objectives[0], ind.objectives[1]))
        .collect();

    if pts.is_empty() {
        return 0.0;
    }

    // Ordena pelo f1 crescente
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut hv = 0.0;
    let mut prev_f1 = 0.0; // assume 0 como mínimo de f1
    for &(f1, f2) in &pts {
        let width = (f1 - prev_f1).max(0.0);
        let height = (reference.1 - f2).max(0.0);
        hv += width * height;
        prev_f1 = f1;
    }

    // Adiciona retângulo final até o ponto de referência
    let last_f1 = pts.last().unwrap().0;
    let last_f2 = pts.last().unwrap().1;
    hv += (reference.0 - last_f1).max(0.0) * (reference.1 - last_f2).max(0.0);

    hv
}

pub fn plot_metric_comparativo(
    values1: &[f64],
    values2: &[f64],
    label1: &str,
    label2: &str,
    metric_name: &str,
    filename: &str,
) {
    let root = BitMapBackend::new(filename, (900, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let min_val = values1.iter().chain(values2.iter()).cloned().fold(f64::INFINITY, f64::min);
    let max_val = values1.iter().chain(values2.iter()).cloned().fold(f64::NEG_INFINITY, f64::max);
    let x_range = 0..values1.len();
    let y_range = (min_val * 0.95)..(max_val * 1.05);

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("{} ao longo das gerações", metric_name), ("sans-serif", 28))
        .margin(40)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(x_range, y_range)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Geração")
        .y_desc(metric_name)
        .label_style(("sans-serif", 16))
        .draw()
        .unwrap();

    // Linha sólida = valores1
    chart
        .draw_series(LineSeries::new(
            values1.iter().enumerate().map(|(i, &v)| (i, v)),
            &BLUE,
        ))
        .unwrap()
        .label(label1)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Linha sólida (ou cor diferente) = valores2
    chart
        .draw_series(LineSeries::new(
            values2.iter().enumerate().map(|(i, &v)| (i, v)),
            &RED,
        ))
        .unwrap()
        .label(label2)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .label_font(("sans-serif", 18))
        .draw()
        .unwrap();
}

pub fn zdt1_discreta(x: &[i32]) -> (f64, f64) {
    let n = x.len();
    assert!(n >= 1, "vetor deve ter pelo menos 1 variável");

    let f1 = x[0] as f64 / 1000.0;

    let g = if n == 1 {
        1.0
    } else {
        let sum_tail: f64 = x.iter().skip(1).map(|&xi| xi as f64 / 1000.0).sum();
        1.0 + 9.0 * (sum_tail / ((n - 1) as f64))
    };

    let ratio = f1 / g;
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

pub fn zdt3_discreta(x: &[i32]) -> (f64, f64) {
    let n = x.len();
    assert!(n >= 1, "vetor deve ter pelo menos 1 variável");

    let f1 = x[0] as f64 / 1000.0;

    let g = if n == 1 {
        1.0
    } else {
        let sum_tail: f64 = x.iter().skip(1).map(|&xi| xi as f64 / 1000.0).sum();
        1.0 + 9.0 * (sum_tail / ((n - 1) as f64))
    };

    let f2 = g * (1.0 - (f1 / g).sqrt() - (f1 / g) * (10.0 * std::f64::consts::PI * f1).sin());
    (f1, f2)
}

pub fn evaluate_individual_zdt3(genes: &BitVec, bit_length: u32) -> Vec<f64> {
    let n_vars = (genes.len() as u32) / bit_length;
    let mut x_vals: Vec<i32> = Vec::with_capacity(n_vars as usize);

    for i in 0..n_vars {
        let start = (i * bit_length) as usize;
        let end = ((i + 1) * bit_length) as usize;
        let slice = &genes[start..end];
        let val = gen_to_fen(slice, 0, 1000, bit_length).round() as i32;
        x_vals.push(val);
    }

    let (f1, f2) = zdt3_discreta(&x_vals);
    vec![f1, f2]
}

pub fn run_zdt1(
    pop_size: usize,
    dim: usize,
    bit_length: usize,
    gens: usize,
    runs: usize,
    crossover_prob: f64,
    mutation_prob: f64,
) {
    for run in 0..runs {
        println!("================== Execução {} ==================", run + 1);
        let mut spacing_vals = Vec::new();
        let mut hypervolume_vals = Vec::new();
        let mut spacing_vals_nc = Vec::new();
        let mut hypervolume_vals_nc = Vec::new();

        let total_bits = dim * bit_length;

        let population = generate_population(pop_size, RepresentationType::Binary { dim: total_bits });

        let mut evaluated: Vec<IndivMO> = population
            .par_iter()
            .map(|repr| {
                if let Representation::Binary(bits) = repr {
                    let objectives = evaluate_individual(bits, bit_length as u32);
                    IndivMO::new(bits.clone(), objectives)
                } else {
                    panic!("População não é binária");
                }
            })
            .collect();

        let mut evaluated_no_crowd: Vec<IndivMO> = population
            .par_iter()
            .map(|repr| {
                if let Representation::Binary(bits) = repr {
                    let objectives = evaluate_individual(bits, bit_length as u32);
                    IndivMO::new(bits.clone(), objectives)
                } else {
                    panic!("População não é binária");
                }
            })
            .collect();

        for g in 0..gens {
            let offspring = make_offspring(&evaluated, pop_size, crossover_prob, mutation_prob, bit_length as u32, evaluate_individual);
            evaluated = nsga2_select(pop_size, &evaluated, &offspring);

            let spacing_val = spacing(&evaluated);
            let hv_val = hypervolume(&evaluated, (1.1, 1.1)); 
            spacing_vals.push(spacing_val);
            hypervolume_vals.push(hv_val);

            
            let offspring_no_crowd = make_offspring(&evaluated_no_crowd, pop_size, crossover_prob, mutation_prob, bit_length as u32, evaluate_individual);
            evaluated_no_crowd = nsga2_select(pop_size, &evaluated_no_crowd, &offspring_no_crowd);

            let spacing_val_nc = spacing(&evaluated_no_crowd);
            let hv_val_nc = hypervolume(&evaluated_no_crowd, (1.1, 1.1)); 
            spacing_vals_nc.push(spacing_val_nc);
            hypervolume_vals_nc.push(hv_val_nc);

            if g % 10 == 0 || g == gens - 1 {
                println!("Run {} - Geração {}/{}", run+1, g+1, gens);
            }
        }

        let filename = format!("zdt1-run-{}.png", run + 1);
        plot_pareto_levels(&evaluated, &filename, gens, pop_size, dim);

        let filename = format!("zdt1_no_crowd-run-{}.png", run + 1);
        plot_pareto_levels(&evaluated_no_crowd, &filename, gens, pop_size, dim);

        plot_metric_comparativo(
            &spacing_vals,
            &spacing_vals_nc,
            "Com crowding",
            "Sem crowding",
            "Spacing",
            &format!("{}_spacing_comp.png", filename)
        );

        plot_metric_comparativo(
            &hypervolume_vals,
            &hypervolume_vals_nc,
            "Com crowding",
            "Sem crowding",
            "Hypervolume",
            &format!("{}_hypervolume_comp.png", filename)
        );

        let csv_filename = format!("zdt{}_run{}_metrics.csv", if filename.contains("zdt1") {1} else {3}, run + 1);
        let mut wtr = File::create(&csv_filename).unwrap();
        wtr.write_all(b"Generation,Spacing,Crowding_Spacing,Hypervolume,Crowding_Hypervolume\n").unwrap();

        for g in 0..gens {
            let line = format!("{},{},{},{},{}\n",
                g+1,
                spacing_vals[g],
                spacing_vals_nc[g],
                hypervolume_vals[g],
                hypervolume_vals_nc[g]
            );
            wtr.write_all(line.as_bytes()).unwrap();
        }

        println!("✅ CSV salvo em '{}'", csv_filename);
    }
}

pub fn run_zdt3(
    pop_size: usize,
    dim: usize,
    bit_length: usize,
    gens: usize,
    runs: usize,
    crossover_prob: f64,
    mutation_prob: f64,
) {
    for run in 0..runs {
        println!("================== Execução {} ==================", run + 1);
        let mut spacing_vals = Vec::new();
        let mut hypervolume_vals = Vec::new();
        let mut spacing_vals_nc = Vec::new();
        let mut hypervolume_vals_nc = Vec::new();

        let total_bits = dim * bit_length;

        let population = generate_population(pop_size, RepresentationType::Binary { dim: total_bits });

        let mut evaluated: Vec<IndivMO> = population
            .par_iter()
            .map(|repr| {
                if let Representation::Binary(bits) = repr {
                    let objectives = evaluate_individual_zdt3(bits, bit_length as u32);
                    IndivMO::new(bits.clone(), objectives)
                } else {
                    panic!("População não é binária");
                }
            })
            .collect();

        let mut evaluated_no_crowd: Vec<IndivMO> = population
            .par_iter()
            .map(|repr| {
                if let Representation::Binary(bits) = repr {
                    let objectives = evaluate_individual_zdt3(bits, bit_length as u32);
                    IndivMO::new(bits.clone(), objectives)
                } else {
                    panic!("População não é binária");
                }
            })
            .collect();

        for g in 0..gens {
            let offspring = make_offspring(&evaluated, pop_size, crossover_prob, mutation_prob, bit_length as u32, evaluate_individual_zdt3);
            evaluated = nsga2_select(pop_size, &evaluated, &offspring);

            let spacing_val = spacing(&evaluated);
            let hv_val = hypervolume(&evaluated, (1.1, 1.1)); 
            spacing_vals.push(spacing_val);
            hypervolume_vals.push(hv_val);
            
            let offspring_no_crowd = make_offspring(&evaluated_no_crowd, pop_size, crossover_prob, mutation_prob, bit_length as u32, evaluate_individual_zdt3);
            evaluated_no_crowd = nsga2_select(pop_size, &evaluated_no_crowd, &offspring_no_crowd);

            let spacing_val_nc = spacing(&evaluated_no_crowd);
            let hv_val_nc = hypervolume(&evaluated_no_crowd, (1.1, 1.1)); 
            spacing_vals_nc.push(spacing_val_nc);
            hypervolume_vals_nc.push(hv_val_nc);

            if g % 10 == 0 || g == gens - 1 {
                println!("Run {} - Geração {}/{}", run+1, g+1, gens);
            }
        }

        let filename = format!("zdt3-run-{}.png", run + 1);
        plot_pareto_levels(&evaluated, &filename, gens, pop_size, dim);
        let filename = format!("zdt3_no_crowd-run-{}.png", run + 1);
        plot_pareto_levels(&evaluated_no_crowd, &filename, gens, pop_size, dim);

        plot_metric_comparativo(
            &spacing_vals,
            &spacing_vals_nc,
            "Com crowding",
            "Sem crowding",
            "Spacing",
            &format!("{}_spacing_comp.png", filename)
        );

        plot_metric_comparativo(
            &hypervolume_vals,
            &hypervolume_vals_nc,
            "Com crowding",
            "Sem crowding",
            "Hypervolume",
            &format!("{}_hypervolume_comp.png", filename)
        );

        let csv_filename = format!("zdt{}_run{}_metrics.csv", if filename.contains("zdt1") {1} else {3}, run + 1);
        let mut wtr = File::create(&csv_filename).unwrap();
        wtr.write_all(b"Generation,Spacing,Crowding_Spacing,Hypervolume,Crowding_Hypervolume\n").unwrap();

        for g in 0..gens {
            let line = format!("{},{},{},{},{}\n",
                g+1,
                spacing_vals[g],
                spacing_vals_nc[g],
                hypervolume_vals[g],
                hypervolume_vals_nc[g]
            );
            wtr.write_all(line.as_bytes()).unwrap();
        }

        println!("✅ CSV salvo em '{}'", csv_filename);
    }
}
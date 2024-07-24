use anyhow::Context;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::StudentsT;

pub fn compute_neg_log_pvalue(t_statistic: f32, degrees_of_freedom: i32) -> f32 {
    let t = t_statistic as f64;
    let dof = degrees_of_freedom as f64;

    match t {
        f if f.is_nan() => f32::NAN,
        f if f.is_infinite() => f32::INFINITY,
        _ => {
            if dof <= 1.0 {
                return f32::NAN;
            }
            let t_dist = StudentsT::new(0.0, 1.0, dof)
                .with_context(|| format!("Failed to compute t-statistic for dof {}", dof))
                .unwrap();
            let p = 2.0 * t_dist.cdf(-t.abs());
            -p.log10() as f32
        }
    }
}

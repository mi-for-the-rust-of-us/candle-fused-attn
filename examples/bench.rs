// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
// EXPLICIT: a benchmark — a panic costs nothing but a re-run.

//! Wall time of the fused attention, forward and forward + backward, at one shape (default the
//! canvas trainer's: b 64, h 6, s 240, d 64), against the composed attention of the same math
//! (matmul, softmax from elementary ops, matmul — autograd on stock candle). GPU.
//!
//! Usage: `cargo run --release --features cuda --example bench [b h s]`

use candle_core::{D, Device, Tensor, Var};
use candle_fused_attn::fused_attention;
use std::time::Instant;

fn composed(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Tensor {
    let scores = (q
        .contiguous()
        .unwrap()
        .matmul(&k.contiguous().unwrap().t().unwrap())
        .unwrap()
        * scale)
        .unwrap();
    let max = scores.max_keepdim(D::Minus1).unwrap();
    let e = scores.broadcast_sub(&max).unwrap().exp().unwrap();
    let p = e.broadcast_div(&e.sum_keepdim(D::Minus1).unwrap()).unwrap();
    p.matmul(&v.contiguous().unwrap()).unwrap()
}

fn main() {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .map(|x| x.parse().unwrap())
        .collect();
    let (b, h, s, d) = (
        *a.first().unwrap_or(&64),
        *a.get(1).unwrap_or(&6),
        *a.get(2).unwrap_or(&240),
        64,
    );
    let dev = Device::new_cuda(0).unwrap();
    let qkv =
        Var::from_tensor(&Tensor::randn(0f32, 1.0, (b, s, 3 * h * d), &dev).unwrap()).unwrap();
    let split = |i: usize| {
        qkv.as_tensor()
            .narrow(2, i * h * d, h * d)
            .unwrap()
            .reshape((b, s, h, d))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
    };
    let (q, k, v) = (split(0), split(1), split(2));
    let scale = 1.0 / (d as f64).sqrt();
    let merge = |o: Tensor| o.transpose(1, 2).unwrap().reshape((b, s, h * d)).unwrap();

    let fused = || merge(fused_attention(&q, &k, &v, scale as f32, false).unwrap());
    let comp = || merge(composed(&q, &k, &v, scale));
    println!("b {b}  h {h}  s {s}  d {d}  (ms per call, median of 20 after 5 warm-up)");
    for (name, f) in [
        ("fused", &fused as &dyn Fn() -> Tensor),
        ("composed", &comp),
    ] {
        let time = |work: &dyn Fn()| {
            let mut t: Vec<f64> = (0..25)
                .map(|_| {
                    dev.synchronize().unwrap();
                    let t0 = Instant::now();
                    work();
                    dev.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() * 1e3
                })
                .skip(5)
                .collect();
            t.sort_by(f64::total_cmp);
            t[t.len() / 2]
        };
        let fwd = time(&|| {
            f();
        });
        let both = time(&|| {
            f().sum_all().unwrap().backward().unwrap();
        });
        println!("{name:>9}: forward {fwd:7.3}   forward+backward {both:7.3}");
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compiles `kernels/*.cu` to PTX with the same tool and flags as candle-kernels, when the
//! `cuda` feature is on. The PTX is loaded at runtime into candle's own CUDA context.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels/fused_attn.cu");
    // Test switch: compile the kernels' ordinary-load fallback (the path of cards below compute
    // capability 8.0) on whatever card builds them, so that path can be tested where CUDA 13 can no
    // longer target those cards (devlog FA7).
    println!("cargo::rerun-if-env-changed=CANDLE_FUSED_ATTN_SYNC_LOADS");
    #[cfg(feature = "cuda")]
    {
        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());
        let mut builder = cudaforge::KernelBuilder::new()
            .source_files(["kernels/fused_attn.cu"])
            .arg("-std=c++17")
            .arg("-O3");
        if std::env::var_os("CANDLE_FUSED_ATTN_SYNC_LOADS").is_some() {
            builder = builder.arg("-DFATTN_SYNC_LOADS");
        }
        let ptx = builder.build_ptx();
        match ptx.and_then(|p| p.write(out_dir.join("ptx.rs"))) {
            Ok(()) => {}
            Err(e) => {
                println!("cargo::error=candle-fused-attn: PTX build failed: {e}");
            }
        }
    }
}

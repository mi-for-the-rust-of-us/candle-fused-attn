// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compiles `kernels/*.cu` to PTX with the same tool and flags as candle-kernels, when the
//! `cuda` feature is on. The PTX is loaded at runtime into candle's own CUDA context.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels/fused_attn.cu");
    #[cfg(feature = "cuda")]
    {
        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());
        let ptx = cudaforge::KernelBuilder::new()
            .source_files(["kernels/fused_attn.cu"])
            .arg("-std=c++17")
            .arg("-O3")
            .build_ptx();
        match ptx.and_then(|p| p.write(out_dir.join("ptx.rs"))) {
            Ok(()) => {}
            Err(e) => {
                println!("cargo::error=candle-fused-attn: PTX build failed: {e}");
            }
        }
    }
}

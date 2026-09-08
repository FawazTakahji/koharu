use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use strum::EnumProperty;

use crate::{
    Hardware, RuntimeConfig, Store, TorchSource, download,
    runtime::{
        DiscoverablePackage, Package, RuntimePackage,
        graph::Component,
        loader,
        packages::{Cuda, Rocm},
        sealed,
    },
    source::extract,
};

const RELEASE: &str = "v2.13.0.5";
const OFFICIAL_VERSION: &str = "2.13.0";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, strum::Display, strum::EnumProperty)]
pub enum Torch {
    // Keep macOS root-first so dyld resolves LibTorch's weak C++ symbols inside
    // one private RTLD_LOCAL image group instead of publishing them globally.
    #[cfg_attr(target_os = "macos", strum(serialize = "metal"))]
    #[cfg_attr(not(target_os = "macos"), strum(serialize = "cpu"))]
    #[strum(props(
        windows = "libiomp5md.dll,c10.dll,torch_global_deps.dll,torch_cpu.dll,torch.dll,koharu-torch.dll",
        linux = "libgomp.so.1,libc10.so,libtorch_global_deps.so,libtorch_cpu.so,libtorch.so,libkoharu-torch.so",
        macos = "libtorch.dylib,libtorch_global_deps.dylib,libtorch_cpu.dylib,libc10.dylib,libkoharu-torch.dylib",
        official_windows = "libiomp5md.dll,libiompstubs5md.dll,uv.dll,c10.dll,torch_global_deps.dll,torch_cpu.dll,shm.dll,torch.dll",
        official_linux = "libgomp.so.1,libc10.so,libshm.so,libtorch_global_deps.so,libtorch_cpu.so,libtorch.so",
        official_macos = "libtorch.dylib,libshm.dylib,libtorch_global_deps.dylib,libtorch_cpu.dylib,libc10.dylib,libomp.dylib"
    ))]
    Cpu,
    #[strum(
        serialize = "cuda",
        props(
            windows = "c10.dll,c10_cuda.dll,caffe2_nvrtc.dll,torch_global_deps.dll,torch_cpu.dll,torch_cuda.dll,torch.dll,koharu-torch.dll",
            linux = "libc10.so,libc10_cuda.so,libcaffe2_nvrtc.so,libtorch_global_deps.so,libtorch_cpu.so,libtorch_cuda.so,libtorch.so,libkoharu-torch.so",
            official_windows = "libiomp5md.dll,libiompstubs5md.dll,zlibwapi.dll,uv.dll,c10.dll,c10_cuda.dll,caffe2_nvrtc.dll,torch_global_deps.dll,torch_cpu.dll,torch_cuda.dll,shm.dll,torch.dll",
            official_linux = "libgomp.so.1,libc10.so,libc10_cuda.so,libcaffe2_nvrtc.so,libshm.so,libtorch_global_deps.so,libtorch_cpu.so,libtorch_cuda.so,libtorch_nvshmem.so,libtorch_cuda_linalg.so,libtorch.so"
        )
    )]
    Cuda,
    #[strum(
        serialize = "hip",
        props(
            windows = "c10.dll,c10_hip.dll,caffe2_nvrtc.dll,torch_global_deps.dll,torch_cpu.dll,torch_hip.dll,torch.dll,koharu-torch.dll",
            linux = "libc10.so,libc10_hip.so,libcaffe2_nvrtc.so,libtorch_global_deps.so,libtorch_cpu.so,libtorch_hip.so,libtorch.so,libkoharu-torch.so"
        )
    )]
    Rocm,
}

impl Torch {
    pub fn library_names(self, source: TorchSource) -> Result<impl Iterator<Item = &'static str>> {
        let (property, prefix) = match source {
            TorchSource::Bundled => (self.platform_property()?, ""),
            TorchSource::Official => (self.platform_property()?, "official_"),
        };
        let property = format!("{prefix}{property}");
        Ok(self
            .get_str(&property)
            .with_context(|| format!("Torch {self} has no {property} libraries"))?
            .split(','))
    }

    fn platform_property(&self) -> Result<&'static str> {
        if cfg!(target_os = "windows") {
            Ok("windows")
        } else if cfg!(target_os = "linux") {
            Ok("linux")
        } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Ok("macos")
        } else {
            anyhow::bail!("Torch does not support this target")
        }
    }

    fn complete(self, root: &Path, source: TorchSource) -> bool {
        let directory = match source {
            TorchSource::Bundled => root.to_owned(),
            TorchSource::Official => root.join("libtorch/lib"),
        };
        self.library_names(source)
            .is_ok_and(|names| names.into_iter().all(|name| directory.join(name).is_file()))
            && (source != TorchSource::Official
                || self != Self::Cpu
                || root
                    .join("libtorch/include/torch/csrc/api/include/torch/torch.h")
                    .is_file())
    }

    fn official_urls(self) -> Result<Vec<String>> {
        let backend = match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cu130",
            Self::Rocm => anyhow::bail!("official Torch wheels are not published for ROCm"),
        };
        if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
            Ok(vec![format!(
                "https://download.pytorch.org/whl/{backend}/torch-{OFFICIAL_VERSION}%2B{backend}-cp312-cp312-win_amd64.whl"
            )])
        } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            Ok(vec![format!(
                "https://download.pytorch.org/whl/{backend}/torch-{OFFICIAL_VERSION}%2B{backend}-cp312-cp312-manylinux_2_28_x86_64.whl"
            )])
        } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
            Ok(vec![format!(
                "https://download.pytorch.org/whl/{backend}/torch-{OFFICIAL_VERSION}%2B{backend}-cp312-cp312-manylinux_2_28_aarch64.whl"
            )])
        } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) && self == Self::Cpu {
            Ok(vec![format!(
                "https://download.pytorch.org/whl/cpu/torch-{OFFICIAL_VERSION}-cp312-cp312-macosx_14_0_arm64.whl"
            )])
        } else {
            anyhow::bail!("Torch {self} has no official wheel for this target")
        }
    }

    fn source(self) -> Result<TorchSource> {
        let source = RuntimeConfig::shared()?.torch_source;
        match (self, source) {
            (Self::Rocm, TorchSource::Official) => {
                tracing::warn!(
                    "official Torch builds are not published for ROCm; using the bundled package"
                );
                Ok(TorchSource::Bundled)
            }
            _ => Ok(source),
        }
    }

    fn asset(self) -> Result<String> {
        let target = if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
            "x86_64-pc-windows-msvc"
        } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            "x86_64-unknown-linux-gnu"
        } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
            "aarch64-unknown-linux-gnu"
        } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            "aarch64-apple-darwin"
        } else {
            anyhow::bail!("Torch {self} does not support this target")
        };
        Ok(format!("{target}-{self}.tar.gz"))
    }

    async fn install_bundled(self) -> Result<PathBuf> {
        let path = Store::root()
            .join("torch")
            .join(RELEASE)
            .join(self.to_string());
        let asset = self.asset()?;

        Store::directory(
            path,
            move |path| self.complete(path, TorchSource::Bundled),
            move |stage| async move {
                let url = format!(
                    "https://github.com/koharu-rs/torch/releases/download/{RELEASE}/{asset}"
                );
                let archive = tempfile::Builder::new().suffix(".tar.gz").tempfile()?;
                download::fetch(&url, archive.path()).await?;
                extract(
                    archive.path(),
                    &stage,
                    &["**/*.dll", "**/*.dylib", "**/*.so", "**/*.so.*"],
                )
            },
        )
        .await
    }

    async fn install_official(self) -> Result<PathBuf> {
        let path = Store::root()
            .join("torch-official")
            .join(OFFICIAL_VERSION)
            .join(self.to_string());
        let urls = self.official_urls()?;
        let libraries = self
            .library_names(TorchSource::Official)?
            .collect::<Vec<_>>();
        let mut patterns = libraries
            .iter()
            .map(|name| format!("torch/lib/{name}"))
            .collect::<Vec<_>>();
        if self == Self::Cpu {
            patterns.extend([
                "torch/include/**/*".to_owned(),
                "torch/share/cmake/**/*".to_owned(),
                "torch/lib/*.lib".to_owned(),
            ]);
        }

        Store::directory(
            path,
            move |path| self.complete(path, TorchSource::Official),
            move |stage| async move {
                for url in &urls {
                    let archive = tempfile::Builder::new().suffix(".whl").tempfile()?;
                    download::fetch(url, archive.path()).await?;
                    let selected = patterns.iter().map(String::as_str).collect::<Vec<_>>();
                    extract(archive.path(), &stage, &selected)?;
                }
                std::fs::rename(stage.join("torch"), stage.join("libtorch"))?;
                Ok(())
            },
        )
        .await
    }
}

impl sealed::Sealed for Torch {}

impl Package for Torch {
    async fn install(self) -> Result<PathBuf> {
        self.install_with(self.source()?).await
    }
}

impl Torch {
    pub async fn install_with(self, source: TorchSource) -> Result<PathBuf> {
        match source {
            TorchSource::Bundled => self.install_bundled().await,
            TorchSource::Official => self.install_official().await,
        }
    }
}

impl DiscoverablePackage for Torch {
    fn discover(hardware: &Hardware) -> Option<Self> {
        if hardware.supports_metal() {
            return Some(Self::Cpu);
        }
        if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
            return hardware.supports_cuda().then_some(Self::Cuda);
        }
        if !cfg!(any(
            all(target_os = "windows", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "x86_64")
        )) {
            return None;
        }
        if hardware.supports_cuda() {
            return Some(Self::Cuda);
        }
        if hardware.supports_rocm() && Rocm::discover(hardware).is_ok() {
            return Some(Self::Rocm);
        }
        tracing::warn!("no supported Torch accelerator was discovered; using CPU");
        Some(Self::Cpu)
    }
}

impl RuntimePackage for Torch {
    const NAME: &'static str = "Torch";

    fn dependencies(self, hardware: &Hardware) -> Result<Vec<Component>> {
        match self {
            Self::Cpu => Ok(Vec::new()),
            Self::Rocm => Ok(vec![Component::Rocm(Rocm::discover(hardware)?)]),
            Self::Cuda => {
                let packages = [
                    Cuda::Runtime13,
                    Cuda::JitLink13,
                    Cuda::Rtc13,
                    Cuda::Blas13,
                    Cuda::Fft12,
                    Cuda::Rand10,
                    Cuda::Sparse12,
                    Cuda::Solver12,
                    Cuda::Dnn920,
                ];
                Ok(packages.into_iter().map(Component::Cuda).collect())
            }
        }
    }

    async fn activate(self) -> Result<()> {
        let source = self.source()?;
        tracing::info!("activating Torch {self} ({source:?})");
        let directory = self.install_with(source).await?;
        let library_directory = match source {
            TorchSource::Bundled => directory,
            TorchSource::Official => directory.join("libtorch/lib"),
        };
        for library in self.library_names(source)? {
            loader::load(library_directory.join(library), false)?;
        }
        Ok(())
    }
}

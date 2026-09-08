use std::sync::OnceLock;

use anyhow::Result;
use serde::{Deserialize, Serialize};

static CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, specta::Type)]
#[serde(rename_all = "lowercase")]
pub enum TorchSource {
    #[default]
    Bundled,
    Official,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, specta::Type)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub torch_source: TorchSource,
}

impl RuntimeConfig {
    pub fn load() -> Result<koharu_config::Config<Self>> {
        koharu_config::load("runtime")
    }

    pub(crate) fn shared() -> Result<Self> {
        if let Some(config) = CONFIG.get() {
            return Ok(*config);
        }
        let config = *Self::load()?.read()?;
        Ok(*CONFIG.get_or_init(|| config))
    }
}

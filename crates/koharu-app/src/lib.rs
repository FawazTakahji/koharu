//! Koharu's Tauri-managed application state, commands, and lifecycle.

mod app;
mod commands;
mod mcp;

pub use app::run;
pub use commands::bindings;
pub use mcp::DEFAULT_PORT;

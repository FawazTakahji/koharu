use anyhow::{Context as _, Result};
use koharu_agent::{Control, Host, Tool as AgentTool, ToolCall};
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ErrorCode,
    Implementation, ListToolsResult, PaginatedRequestParams, ProtocolVersion, ServerCapabilities,
    ServerInfo, Tool as McpTool,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler};
use serde_json::json;
use std::borrow::Cow;
use tauri::{AppHandle, Cef};
use uuid::Uuid;

use crate::commands::agent::KoharuHost;

pub const DEFAULT_PORT: u16 = 41313;

/// Protocol versions the MCP server will negotiate, capped below `2026-07-28`.
static SUPPORTED_PROTOCOLS: [ProtocolVersion; 4] = [
    ProtocolVersion::V_2024_11_05,
    ProtocolVersion::V_2025_03_26,
    ProtocolVersion::V_2025_06_18,
    ProtocolVersion::V_2025_11_25,
];

/// Expose Koharu's agent tools over the Model Context Protocol.
///
/// The server binds the Streamable HTTP transport on `127.0.0.1` so only
/// local agents can reach it. Tool invocations run through the same
/// [`KoharuHost`] the in-app agent uses, so MCP requests share project,
/// processing, and cancellation behavior with the UI.
#[derive(Clone)]
pub(crate) struct KoharuMcp {
    host: KoharuHost,
}

impl KoharuMcp {
    pub(crate) fn new(handle: AppHandle<Cef>) -> Self {
        Self {
            host: KoharuHost::new(handle),
        }
    }
}

impl ServerHandler for KoharuMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::V_2025_11_25)
            .with_server_info(Implementation::new("koharu", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Koharu is an editor for translated typesetting. Inspect the open project with \
                 inspect_project, render pages with view_page, then edit text, typography, \
                 geometry, and visibility. Run the processing pipeline stages (detection, OCR, \
                 translation, inpainting) with run_pipeline.",
            )
    }

    fn supported_protocol_versions(&self) -> Cow<'static, [ProtocolVersion]> {
        // SEP-2549 requires `ttlMs`/`cacheScope` on results for peers negotiating
        // `2026-07-28`, which we intentionally do not implement; cap negotiation
        // so such peers fall back to `2025-11-25`.
        Cow::Borrowed(&SUPPORTED_PROTOCOLS)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let tools = self.host.tools().into_iter().map(mcp_tool).collect();
        Ok(ListToolsResult::with_all_items(tools))
    }

    fn get_tool(&self, name: &str) -> Option<McpTool> {
        self.host
            .tools()
            .into_iter()
            .find(|tool| tool.name == name)
            .map(mcp_tool)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let name = request.name.as_ref();
        let Some(_) = self.host.tools().into_iter().find(|tool| tool.name == name) else {
            return Err(McpError::new(
                ErrorCode::METHOD_NOT_FOUND,
                format!("no such tool: {name}"),
                Some(json!({ "tool": name })),
            ));
        };
        let call = ToolCall {
            call_id: Uuid::new_v4().to_string(),
            name: name.to_owned(),
            arguments: request
                .arguments
                .map(|arguments| serde_json::to_string(&arguments).unwrap_or_default())
                .unwrap_or_else(|| "{}".to_owned()),
        };
        let control = Control::default();
        let watcher = tauri::async_runtime::spawn({
            let control = control.clone();
            let cancelled = context.ct.clone();
            async move {
                cancelled.cancelled().await;
                control.cancel();
            }
        });
        let invocation = match self.host.invoke(call, &control).await {
            Ok(invocation) => invocation,
            Err(error) => {
                return Ok(
                    CallToolResult::error(vec![ContentBlock::text(error.to_string())]).into(),
                );
            }
        };
        watcher.abort();
        let mut content = vec![ContentBlock::text(
            serde_json::to_string_pretty(&invocation.value)
                .unwrap_or_else(|_| invocation.value.to_string()),
        )];
        for image in invocation.images {
            if let Some((mime, data)) = split_data_url(&image.data_url) {
                content.push(ContentBlock::image(data, mime));
            }
        }
        Ok(CallToolResult::success(content).into())
    }
}

/// Run the MCP server until the application exits. A port of `0` disables it.
pub(crate) async fn serve(handle: AppHandle<Cef>, port: u16) -> Result<()> {
    if port == 0 {
        tracing::info!("MCP server is disabled");
        return Ok(());
    }
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .with_context(|| format!("failed to bind the MCP server on 127.0.0.1:{port}"))?;
    let service = StreamableHttpService::new(
        move || Ok(KoharuMcp::new(handle.clone())),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    tracing::info!(
        url = format!("http://127.0.0.1:{port}/mcp"),
        "MCP server is listening",
    );
    axum::serve(listener, router)
        .await
        .context("the MCP server stopped unexpectedly")
}

fn mcp_tool(tool: AgentTool) -> McpTool {
    let schema = tool.parameters.as_object().cloned().unwrap_or_default();
    McpTool::new(tool.name, tool.description, schema)
}

/// Split a `data:<mime>;base64,<payload>` URL into its parts.
fn split_data_url(data_url: &str) -> Option<(String, String)> {
    let (meta, data) = data_url.split_once(',')?;
    let mime = meta
        .strip_prefix("data:")
        .and_then(|meta| meta.split(';').next())
        .filter(|mime| !mime.is_empty())
        .unwrap_or("image/webp");
    Some((mime.to_owned(), data.to_owned()))
}

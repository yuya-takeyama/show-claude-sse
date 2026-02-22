use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::fs;
use std::io::{self, Read};

/// Parse and display Anthropic Messages API SSE streaming responses
#[derive(Parser)]
#[command(name = "show-claude-sse")]
#[command(about = "Parse and display Anthropic Messages API SSE streaming responses")]
struct Cli {
    /// Input file path (use '-' or omit for stdin)
    file: Option<String>,

    /// Output raw JSON for each event
    #[arg(long)]
    json: bool,

    /// Output only the reconstructed text content
    #[arg(long)]
    content_only: bool,
}

// ===== Data Structures =====

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum SseEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageInfo },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: Delta },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop {
        #[allow(dead_code)]
        index: usize,
    },

    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaData,
        usage: Option<Usage>,
        #[serde(default)]
        #[allow(dead_code)]
        context_management: Option<serde_json::Value>,
    },

    #[serde(rename = "message_stop")]
    MessageStop,

    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "error")]
    Error { error: ErrorInfo },
}

#[derive(Debug, Deserialize, Clone)]
struct MessageInfo {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    stop_sequence: Option<String>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaData {
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    stop_sequence: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: String,
    },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[allow(dead_code)]
        input: serde_json::Value,
    },

    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        #[allow(dead_code)]
        id: String,
        #[allow(dead_code)]
        name: String,
        #[allow(dead_code)]
        input: serde_json::Value,
    },

    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult {
        #[allow(dead_code)]
        tool_use_id: String,
        #[allow(dead_code)]
        content: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum Delta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },

    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },

    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },

    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

#[derive(Debug, Deserialize, Clone, Default)]
struct Usage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    cache_creation: Option<CacheCreation>,
    #[serde(default)]
    #[allow(dead_code)]
    server_tool_use: Option<serde_json::Value>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    inference_geo: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct CacheCreation {
    #[serde(default)]
    ephemeral_5m_input_tokens: Option<u64>,
    #[serde(default)]
    ephemeral_1h_input_tokens: Option<u64>,
}

#[derive(Debug, Deserialize, Clone)]
struct ErrorInfo {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

// ===== SSE Parser =====

#[derive(Debug)]
struct RawEvent {
    event_type: String,
    data: String,
}

fn parse_sse(input: &str) -> Vec<RawEvent> {
    let mut events = Vec::new();
    let mut current_event_type = String::new();
    let mut current_data_lines: Vec<String> = Vec::new();

    for line in input.lines() {
        if let Some(stripped) = line.strip_prefix("event:") {
            current_event_type = stripped.trim().to_string();
        } else if let Some(stripped) = line.strip_prefix("data:") {
            current_data_lines.push(stripped.trim().to_string());
        } else if line.trim().is_empty() {
            // Empty line = event delimiter
            if !current_data_lines.is_empty() {
                let data = current_data_lines.join("\n");
                events.push(RawEvent {
                    event_type: current_event_type.clone(),
                    data,
                });
                current_data_lines.clear();
            }
            current_event_type.clear();
        }
    }

    // Handle case where file doesn't end with an empty line
    if !current_data_lines.is_empty() {
        let data = current_data_lines.join("\n");
        events.push(RawEvent {
            event_type: current_event_type,
            data,
        });
    }

    events
}

// ===== State Accumulator =====

#[derive(Debug)]
enum AccumulatedBlock {
    Text(String),
    Thinking { content: String, signature: String },
    ToolUse {
        id: String,
        name: String,
        accumulated_json: String,
    },
}

#[derive(Debug, Default)]
struct MessageAccumulator {
    model: Option<String>,
    message_id: Option<String>,
    role: Option<String>,
    stop_reason: Option<String>,
    service_tier: Option<String>,
    inference_geo: Option<String>,
    content_blocks: Vec<AccumulatedBlock>,
    initial_usage: Option<Usage>,
    final_usage: Option<Usage>,
    errors: Vec<ErrorInfo>,
}

impl MessageAccumulator {
    fn new() -> Self {
        Self::default()
    }

    fn process_event(&mut self, event: &SseEvent) {
        match event {
            SseEvent::MessageStart { message } => {
                self.model = message.model.clone();
                self.message_id = message.id.clone();
                self.role = message.role.clone();
                self.stop_reason = message.stop_reason.clone();
                if let Some(usage) = &message.usage {
                    self.service_tier = usage.service_tier.clone();
                    self.inference_geo = usage.inference_geo.clone();
                    self.initial_usage = Some(usage.clone());
                }
            }
            SseEvent::ContentBlockStart { index, content_block } => {
                // Ensure vector is large enough
                while self.content_blocks.len() <= *index {
                    self.content_blocks
                        .push(AccumulatedBlock::Text(String::new()));
                }
                match content_block {
                    ContentBlock::Text { text } => {
                        self.content_blocks[*index] = AccumulatedBlock::Text(text.clone());
                    }
                    ContentBlock::Thinking { thinking, signature } => {
                        self.content_blocks[*index] = AccumulatedBlock::Thinking {
                            content: thinking.clone(),
                            signature: signature.clone(),
                        };
                    }
                    ContentBlock::ToolUse { id, name, .. } => {
                        self.content_blocks[*index] = AccumulatedBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            accumulated_json: String::new(),
                        };
                    }
                    _ => {}
                }
            }
            SseEvent::ContentBlockDelta { index, delta } => {
                if let Some(block) = self.content_blocks.get_mut(*index) {
                    match (block, delta) {
                        (AccumulatedBlock::Text(s), Delta::TextDelta { text }) => {
                            s.push_str(text);
                        }
                        (
                            AccumulatedBlock::Thinking {
                                content,
                                signature: _sig,
                            },
                            Delta::ThinkingDelta { thinking },
                        ) => {
                            content.push_str(thinking);
                        }
                        (
                            AccumulatedBlock::Thinking {
                                signature: sig, ..
                            },
                            Delta::SignatureDelta { signature },
                        ) => {
                            sig.push_str(signature);
                        }
                        (
                            AccumulatedBlock::ToolUse {
                                accumulated_json, ..
                            },
                            Delta::InputJsonDelta { partial_json },
                        ) => {
                            accumulated_json.push_str(partial_json);
                        }
                        _ => {}
                    }
                }
            }
            SseEvent::MessageDelta { delta, usage, .. } => {
                if let Some(reason) = &delta.stop_reason {
                    self.stop_reason = Some(reason.clone());
                }
                if let Some(u) = usage {
                    self.final_usage = Some(u.clone());
                }
            }
            SseEvent::Error { error } => {
                self.errors.push(error.clone());
            }
            _ => {} // Ping, ContentBlockStop, MessageStop -- no state changes needed
        }
    }
}

// ===== Output Formatters =====

fn print_formatted(acc: &MessageAccumulator) {
    println!("=== Claude SSE Response ===");
    println!();
    if let Some(model) = &acc.model {
        println!("Model: {}", model);
    }
    if let Some(id) = &acc.message_id {
        println!("Message ID: {}", id);
    }
    if let Some(reason) = &acc.stop_reason {
        println!("Stop Reason: {}", reason);
    }
    if let Some(tier) = &acc.service_tier {
        println!("Service Tier: {}", tier);
    }
    if let Some(geo) = &acc.inference_geo {
        println!("Inference Geo: {}", geo);
    }
    println!();

    // Print thinking blocks if present
    let has_thinking = acc
        .content_blocks
        .iter()
        .any(|b| matches!(b, AccumulatedBlock::Thinking { .. }));
    if has_thinking {
        println!("--- Thinking ---");
        for block in &acc.content_blocks {
            if let AccumulatedBlock::Thinking { content, .. } = block {
                print!("{}", content);
            }
        }
        println!();
        println!("---");
        println!();
    }

    // Print text content
    println!("--- Content ---");
    for block in &acc.content_blocks {
        if let AccumulatedBlock::Text(text) = block {
            print!("{}", text);
        }
    }
    println!();
    println!("---");

    // Print tool use blocks if present
    let has_tool_use = acc
        .content_blocks
        .iter()
        .any(|b| matches!(b, AccumulatedBlock::ToolUse { .. }));
    if has_tool_use {
        println!();
        println!("--- Tool Use ---");
        for block in &acc.content_blocks {
            if let AccumulatedBlock::ToolUse {
                id,
                name,
                accumulated_json,
            } = block
            {
                println!("Tool: {} ({})", name, id);
                // Try to pretty-print the JSON input
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(accumulated_json) {
                    println!(
                        "Input: {}",
                        serde_json::to_string_pretty(&parsed).unwrap_or(accumulated_json.clone())
                    );
                } else {
                    println!("Input: {}", accumulated_json);
                }
            }
        }
        println!("---");
    }

    // Print usage -- prefer final_usage (from message_delta, cumulative) over initial_usage
    let usage = acc.final_usage.as_ref().or(acc.initial_usage.as_ref());
    if let Some(u) = usage {
        println!();
        println!("=== Usage ===");
        if let Some(v) = u.input_tokens {
            println!("Input Tokens: {}", v);
        }
        if let Some(v) = u.cache_creation_input_tokens {
            println!("Cache Creation: {}", v);
        }
        if let Some(v) = u.cache_read_input_tokens {
            println!("Cache Read: {}", v);
        }
        if let Some(v) = u.output_tokens {
            println!("Output Tokens: {}", v);
        }
        if let Some(cc) = &u.cache_creation {
            println!();
            println!("Cache Breakdown:");
            if let Some(v) = cc.ephemeral_5m_input_tokens {
                println!("  - Ephemeral 5m: {}", v);
            }
            if let Some(v) = cc.ephemeral_1h_input_tokens {
                println!("  - Ephemeral 1h: {}", v);
            }
        }
    }

    // Print errors if any
    if !acc.errors.is_empty() {
        println!();
        println!("=== Errors ===");
        for err in &acc.errors {
            println!("[{}] {}", err.error_type, err.message);
        }
    }
}

fn print_json(raw_events: &[RawEvent]) -> Result<()> {
    for raw in raw_events {
        let parsed: serde_json::Value = serde_json::from_str(&raw.data)
            .with_context(|| format!("Failed to parse JSON for event '{}'", raw.event_type))?;
        let output = serde_json::json!({
            "event": raw.event_type,
            "data": parsed,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    }
    Ok(())
}

fn print_content_only(acc: &MessageAccumulator) {
    for block in &acc.content_blocks {
        if let AccumulatedBlock::Text(text) = block {
            print!("{}", text);
        }
    }
    // Ensure trailing newline
    println!();
}

// ===== Main Function =====

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Read input
    let input = match cli.file.as_deref() {
        None | Some("-") => {
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .context("Failed to read from stdin")?;
            buf
        }
        Some(path) => fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path))?,
    };

    // Parse SSE events
    let raw_events = parse_sse(&input);

    if raw_events.is_empty() {
        anyhow::bail!("No SSE events found in input");
    }

    // JSON mode: just print each event as JSON
    if cli.json {
        return print_json(&raw_events);
    }

    // Parse and accumulate events
    let mut acc = MessageAccumulator::new();
    for raw in &raw_events {
        match serde_json::from_str::<SseEvent>(&raw.data) {
            Ok(event) => acc.process_event(&event),
            Err(e) => {
                eprintln!(
                    "Warning: Failed to parse event '{}': {}",
                    raw.event_type, e
                );
            }
        }
    }

    // Output based on mode
    if cli.content_only {
        print_content_only(&acc);
    } else {
        print_formatted(&acc);
    }

    Ok(())
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sse_basic() {
        let input = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\",\"model\":\"claude\",\"role\":\"assistant\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null}}\n\nevent: ping\ndata: {\"type\":\"ping\"}\n\n";
        let events = parse_sse(input);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "message_start");
        assert_eq!(events[1].event_type, "ping");
    }

    #[test]
    fn test_parse_sse_no_trailing_newline() {
        let input = "event: ping\ndata: {\"type\":\"ping\"}";
        let events = parse_sse(input);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "ping");
    }

    #[test]
    fn test_accumulator_text_content() {
        let mut acc = MessageAccumulator::new();

        // Simulate message_start
        let event1 = SseEvent::MessageStart {
            message: MessageInfo {
                id: Some("msg_123".to_string()),
                model: Some("claude".to_string()),
                role: Some("assistant".to_string()),
                stop_reason: None,
                stop_sequence: None,
                usage: None,
            },
        };
        acc.process_event(&event1);

        // Content block start
        let event2 = SseEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::Text {
                text: String::new(),
            },
        };
        acc.process_event(&event2);

        // Add text deltas
        let event3 = SseEvent::ContentBlockDelta {
            index: 0,
            delta: Delta::TextDelta {
                text: "Hello".to_string(),
            },
        };
        acc.process_event(&event3);

        let event4 = SseEvent::ContentBlockDelta {
            index: 0,
            delta: Delta::TextDelta {
                text: " world!".to_string(),
            },
        };
        acc.process_event(&event4);

        // Verify content
        assert_eq!(acc.content_blocks.len(), 1);
        match &acc.content_blocks[0] {
            AccumulatedBlock::Text(text) => assert_eq!(text, "Hello world!"),
            _ => panic!("Expected text block"),
        }
    }
}

const API_BASE = "https://mg643-offrails.hf.space";

// ── Good trace — clean, single tool call, resolved ───────────────────────────
export const SAMPLE_TRACE_GOOD = JSON.stringify({
  id: "trace_good",
  conversations: [
    { from: "system", value: "You are a helpful assistant with access to web_search." },
    { from: "user", value: "What is the capital of Australia?" },
    { from: "assistant", value: "Let me look that up." },
    { from: "tool_call", value: JSON.stringify({ name: "web_search", args: { query: "capital of Australia" } }) },
    { from: "observation", value: "The capital of Australia is Canberra." },
    { from: "assistant", value: "The capital of Australia is Canberra." },
  ],
}, null, 2);

// ── Bad trace — circular calls, errors, apologies, give-up language ──────────
export const SAMPLE_TRACE_BAD = JSON.stringify({
  id: "trace_bad",
  conversations: [
    { from: "system", value: "You are a helpful assistant with access to web_search and calculator tools." },
    { from: "user", value: "What is the GDP of France in 2023 and multiply it by 0.03?" },
    { from: "assistant", value: "I'll search for that." },
    { from: "tool_call", value: JSON.stringify({ name: "web_search", args: { query: "France GDP 2023" } }) },
    { from: "observation", value: "Error: API timeout. Unable to retrieve results." },
    { from: "assistant", value: "Sorry, let me try again." },
    { from: "tool_call", value: JSON.stringify({ name: "web_search", args: { query: "France GDP 2023" } }) },
    { from: "observation", value: "Error: rate limit exceeded. Failed to fetch data." },
    { from: "assistant", value: "Unfortunately I cannot retrieve this. Let me try a different query." },
    { from: "tool_call", value: JSON.stringify({ name: "web_search", args: { query: "France GDP 2023" } }) },
    { from: "observation", value: "Error: 503 service unavailable." },
    { from: "assistant", value: "I'm sorry, I give up. I won't be able to answer this question." },
  ],
}, null, 2);

export const SAMPLE_TRACE = SAMPLE_TRACE_GOOD;


// ── Parse raw textarea input into a conversations array ───────────────────────
export function parseTrace(raw) {
  try {
    const parsed = JSON.parse(raw);
    if (parsed.conversations) return parsed.conversations;
    if (Array.isArray(parsed)) return parsed;
    if (parsed.messages) return parsed.messages;
    return null;
  } catch {
    return null;
  }
}


// ── Call the real backend ─────────────────────────────────────────────────────
export async function analyzeTrace(conversations) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversations }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `API error ${response.status}`);
  }

  const data = await response.json();
  const f = data.features || {};

  const toolCalls = f.num_tool_calls ?? 0;
  const uniqueTools = f.unique_tools ?? 0;
  const totalSteps = f.num_turns ?? conversations.length;

  // Correct: fraction of calls that were the most-repeated tool
  const repeatRatio = toolCalls > 0
    ? Math.round(((f.max_single_tool_freq ?? 1) / toolCalls) * 100)
    : 0;

  const signalToFlag = (signal) => {
    const s = signal.toLowerCase();
    if (s.includes("circular") || s.includes("consecutive"))
      return { level: "high", text: signal, tag: "circular · tool overuse" };
    if (s.includes("repetition") || s.includes("duplicate"))
      return { level: "high", text: signal, tag: "tool repetition" };
    if (s.includes("no tool calls"))
      return { level: "medium", text: signal, tag: "missing tool use" };
    if (s.includes("error") || s.includes("empty"))
      return { level: "medium", text: signal, tag: "observation quality" };
    if (s.includes("gave up") || s.includes("give up"))
      return { level: "high", text: signal, tag: "goal abandonment" };
    if (s.includes("apology") || s.includes("failure") || s.includes("sorry"))
      return { level: "medium", text: signal, tag: "failure language" };
    return { level: "low", text: signal, tag: "anomaly signal" };
  };

  const flags = (data.anomaly_signals || []).map(signalToFlag);
  const score = Math.round(data.confidence * 100);

  return { score, flags, isAnomalous: data.is_anomalous, metrics: { steps: totalSteps, toolCalls, uniqueTools, repeatRatio } };
}


export function getRoleClass(role) {
  if (role === "system") return "system";
  if (role === "user") return "user";
  if (role === "assistant") return "assistant";
  if (role === "tool_call" || role === "function_call") return "tool";
  if (role === "observation" || role === "tool_response" || role === "function") return "result";
  return "assistant";
}

export const ROLE_ICONS = {
  system: "⚙", user: "◎", assistant: "◈",
  tool_call: "⬡", function_call: "⬡",
  observation: "◫", tool_response: "◫", error: "✕",
};
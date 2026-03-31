Analyze the input and generate a JSON object.

Input: {input_text}
Soft Prompts (Bridge): {soft_prompts}
System Directives: {system_directives}

# MEMORY & CONTEXT
Semantic Rules: {semantic_rules}
Knowledge Triples: {knowledge_triples}
Recent Episodes (Past interactions):
{recent_episodes}

# RUNTIME OBSERVATIONS (Tool Outputs)
{runtime_observations}

IMPORTANT - MULTI-STEP REASONING:
1. If you need more information, output actions like "execute_bash", "search_web", "read_file", "introspect_self", "call_mcp_tool" or "spawn_subordinate".
2. Leave 'response_text' empty ("") while gathering data.
3. Once the '# RUNTIME OBSERVATIONS' section contains the facts you need, you MUST PROVIDE THE FINAL ANSWER in 'response_text' and stop calling tools.
4. Do not repeat the same tool call if the observation already contains the answer.
5. Synthesize facts into a helpful response.

Available Action Types:
{available_actions}

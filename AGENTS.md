# AGENTS.md

## Project

FitText — evolving agent tool ecologies via memetic retrieval. Agents retrieve,
compose, and execute tools (ToolBench / StableToolBench / Toolret style), so the
code runs model-generated artifacts and calls external tool services.

## Code review — security audit (Codex)

When reviewing a pull request, in addition to correctness and style, perform a
security audit. Report each finding with a severity tag (CRITICAL / HIGH /
MEDIUM / LOW) and a concrete fix. Default to flagging when unsure rather than
assuming a change is safe.

Because this codebase retrieves and executes tools, weight these highest:

- **Arbitrary code / tool execution** — any path that runs retrieved or
  model-generated tool code (`exec`, `eval`, `subprocess`, dynamic import,
  `shell=True`). Require sandboxing/isolation and validated inputs; never run
  untrusted tool code unconstrained in the host environment.
- **Prompt / tool-description injection** — retrieved tool metadata, memes, or
  API responses that flow into prompts or execution decisions are untrusted;
  flag places where they can hijack control flow.
- **Command / shell injection** — `subprocess` / `os.system` / `shell=True`
  built from non-constant input. Require argument lists, not string
  interpolation.
- **Secrets & credential leakage** — LLM/tool API keys, tokens, ToolBench keys
  in code, logs, errors, configs, fixtures, or committed files. No secrets in
  the repo; redact token patterns in logs.
- **SSRF & untrusted fetches** — tool calls / HTTP requests to
  attacker-controllable URLs must validate targets and enforce timeouts.
- **Unsafe deserialization** — no `pickle` / `yaml.load` / `torch.load` of
  untrusted tool caches or downloaded artifacts; use safe loaders.
- **Path traversal & unsafe file IO** — tool/retrieval-supplied paths must be
  confined; no writing outside intended directories.
- **Dependency risk** — new or bumped dependencies (including tool servers):
  flag known-vulnerable versions and unpinned ranges.
- **Insecure defaults** — disabled sandboxing, `verify=False`, debug servers
  bound to `0.0.0.0`, overly broad file/network permissions.

If a change touches none of these surfaces, say so explicitly rather than
staying silent.

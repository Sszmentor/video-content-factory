# AGENTS.md — Shared Rules for All AI Agents
# Applies to: Claude Code, Codex, and any future coding agents
# Location: root of ~/Projects/AI_Agents/ (symlink into each repo as needed)
# Version: 1.0 (2026-03-03)

## Project Structure

```
~/Projects/AI_Agents/          ← git repos (AI_REPOS_ROOT)
~/Dropbox/.../AI_Agents/       ← data, artifacts (AI_DATA_ROOT)
```

Git repos are NOT in Dropbox. Code syncs via GitHub, data syncs via Dropbox.

## Commit Conventions

- Commit messages: English, imperative mood (`fix:`, `feat:`, `chore:`, `refactor:`)
- Always include `closes #N` when resolving a GitHub issue
- Co-author line for AI agents:
  - Claude Code: `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`
  - Codex: `Co-Authored-By: Codex <noreply@openai.com>`

## Branch Naming

- `feat/short-description` — new features
- `fix/short-description` — bug fixes
- `chore/short-description` — maintenance

## Code Style

- Python: follow existing project conventions (each repo may differ)
- No unnecessary refactoring — fix what's asked, don't "improve" surrounding code
- No docstrings/comments unless logic is non-obvious
- Security: never commit `.env`, `.session`, `*_token.json`, `client_secret.json`

## File Naming (Russian content)

- Format: `YY.MM.DD Description (type).md`
- Date = creation date, 2-digit year, dots: `26.03.03`
- Types in parentheses: `(транскрипт)`, `(саммари)`, `(аналитика)`, `(план)`, `(конспект)`, `(пост)`
- Language: Russian for content, English for code

## Domain Rules (DeFi-Hedonist)

- "Студенты" (students), NEVER "ученики" (pupils) — it's a DeFi University
- Em-dash `—` (U+2014) ONLY. Never `–` or `-` as a dash
- Product tiers: 0→1→2→3→4. Never sell same or lower tier

## Coordination

- All tasks tracked in GitHub Projects (board "Agent Tasks", owner Sszmentor)
- One agent per issue at a time — claim by assigning yourself
- PRs require review before merge to main
- If your work conflicts with another agent's PR — comment on the PR, don't force-push

## Secrets

- Never hardcode secrets in code or commits
- Use 1Password references: `op://AI_Agents/ITEM_NAME/credential`
- For CI/CD: GitHub Secrets or Railway env vars
- `.env` files are .gitignored and never committed

## Testing

- Dependencies always via `requirements.txt`, never inline in CI yml
- Run existing tests before pushing: `pytest` or project-specific command
- If no tests exist, at minimum verify the app starts without errors

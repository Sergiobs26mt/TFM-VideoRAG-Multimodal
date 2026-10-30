# Global rules

## Writing style
- Never use em-dashes (`—`). Use a regular hyphen-minus (`-`) instead.
- Write in ASD-STE100 Simplified Technical English: short sentences, active voice, no jargon.

## Git commits
- Never commit without explicit user permission. Always show what would be committed and wait for approval first.
- Never add `Co-Authored-By` trailers to commit messages.

## Coding style
Always preserve the existing coding style when editing code or adding features.

## Comments, docstrings & docs
Write only what a reader can't get from the code, names, types or git history.
- Comments: one line when possible, fine even on fairly obvious code. Say *why* when there is a reason (constraint, workaround, surprising choice). No line-by-line or long explanations.
- Docstrings: one-line summary; more only for non-obvious behavior.
- Describe the present only: no history ("previously", "now uses") and no references to chats or authorship ("as requested", "added by agent").
- Keep specifics when shortening: "lowest AIC or BIC", not "lowest criterion".
- No banners, filler ("Note that") or commented-out code.
- Markdown docs: present tense, verified against the code (commands, paths, env vars, defaults). State each fact once and link to it. No status logs outside changelogs.
- Leave alone: license headers, tool directives (`noqa`, `type: ignore`, `eslint-disable`), changelogs, decision records, TODO/FIXME.
- Agent instruction files (AGENTS.md, skills, rules): one-clause why per rule; drop generic advice the model already follows.
- Remove or update obsolete documentation, comments and docstrings.

## Obsolete code
- Do not preserve backwards compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks or migrations.
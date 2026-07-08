# Codex and ChatGPT PR Workflow

Goal: Codex implements code changes, commits them to Git, opens a GitHub pull request, and writes a handoff note. The user sends the PR link to ChatGPT for review, then merges on GitHub after review.

## Codex Responsibilities

After each implementation task, Codex should:

1. Run the relevant tests or smoke checks.
2. Review the diff with `git diff`.
3. Create or update a handoff document under `docs/handoffs/`.
4. Commit the code and handoff document with a clear message.
5. Push a task branch to GitHub.
6. Create a pull request and give the user the PR link.

## ChatGPT Review Responsibilities

The user sends the PR link to ChatGPT and asks it to review:

- Correctness risks
- Modeling or algorithmic mistakes
- Missing tests
- Reproducibility gaps
- Code clarity and maintainability
- Whether the handoff matches the actual diff

## Required Repository Setup

This workflow requires a GitHub remote and one PR creation method:

```powershell
git remote add origin https://github.com/<owner>/<repo>.git
git push -u origin master
```

Then install and log in to GitHub CLI:

```powershell
gh auth login
```

After that, Codex can create PRs with:

```powershell
git checkout -b codex/<short-task-name>
git push -u origin codex/<short-task-name>
gh pr create --fill
```

## Handoff File Naming

Use:

```text
docs/handoffs/YYYY-MM-DD-short-task-name.md
```

Each handoff should be committed with the code change it describes.

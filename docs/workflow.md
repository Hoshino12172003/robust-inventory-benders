# Development Workflow

This project uses two layers of rollback and review:

1. Git is the formal version-control record. Commit after each meaningful model, algorithm, experiment, or document change.
2. PyCharm Local History is the local safety net. Keep the project open in PyCharm while Codex or PyCharm edits files so external changes are recorded by the IDE file watcher.

Recommended loop:

```powershell
git status --short
git diff
git add src configs tests README.md requirements.txt scripts docs
git commit -m "Describe the research-code change"
```

Before larger changes, create a Git checkpoint:

```powershell
git status --short
git commit -am "Checkpoint before changing Benders strategy"
```

For PyCharm review:

- Use Git tool window for committed and uncommitted diffs.
- Use right click on a file or folder, then Local History, then Show History for IDE-level recovery.
- Use Local History labels manually in PyCharm before risky edits.

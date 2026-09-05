# Second Brain updater

Place `update.ps1` and `.gitignore` in the root of the existing `second_brain_v1` folder.

## One-time setup

After a GitHub repository is prepared for Second Brain:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\update.ps1 -Setup -RepoUrl https://github.com/OWNER/REPO.git
```

## Future updates

```powershell
.\update.ps1
```

Persistent files such as `.env`, databases, and `data/` are excluded from Git updates by `.gitignore`.

`update.ps1` also stashes tracked local code edits temporarily before updating and restores them afterward.

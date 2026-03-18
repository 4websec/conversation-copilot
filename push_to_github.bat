@echo off
cd /d C:\Users\lando\Downloads\conversation-copilot_1\conversation-copilot

echo === Git Status ===
git status --short

echo === Committing ===
git add -A
git commit -m "Initial-commit-conversation-copilot"

echo === Creating GitHub Repo ===
gh repo create conversation-copilot --public --source=. --remote=origin --push

echo === Done ===
git log --oneline -3
pause

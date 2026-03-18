@echo off
cd /d C:\Users\lando\Downloads\conversation-copilot_1\conversation-copilot

echo === GIT STATUS ===
git status --short

echo === GIT LOG ===
git log --oneline -1 2>nul || echo No commits yet

echo === COMMITTING ===
git add -A
git commit -m "Initial commit"

echo === CREATE REPO ===
gh repo create conversation-copilot --private --source=. --push

echo === DONE ===

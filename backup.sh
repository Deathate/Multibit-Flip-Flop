#!/bin/bash

# Strict mode
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error.
set -u
# The return value of a pipeline is the status of the last command to exit with a non-zero status,
# or zero if no command exited with a non-zero status.
set -o pipefail

# cp .gitignore .gitignore.bak
# find * -size +49M | cat >> .gitignore

echo "INFO: Staging all changes..."
git add -A || { echo "ERROR: 'git add -A' failed." >&2; exit 1; }
echo "INFO: All changes staged."

# Check if there are actual changes staged for commit
# `git diff-index --quiet --cached HEAD --` exits 0 if no changes staged, 1 if changes are staged.
if git diff-index --quiet --cached HEAD --; then
    echo "INFO: No changes staged for commit. Nothing further to do."
    # If .gitignore was manipulated, it should be restored here.
    # rm .gitignore
    # mv .gitignore.bak .gitignore
else
    # Using a more standard and sortable date format for commit messages
    COMMIT_MSG="Auto commit $(date +'%Y-%m-%d %H:%M:%S')"
    echo "INFO: Committing staged changes with message: '$COMMIT_MSG'..."
    git commit -m "$COMMIT_MSG" || { echo "ERROR: 'git commit' failed." >&2; exit 1; }
    echo "INFO: Commit successful."

    # If .gitignore was manipulated and commit was successful, restore .gitignore before push
    # rm .gitignore
    # mv .gitignore.bak .gitignore

    echo "INFO: Force pushing to remote..."
    # WARNING: 'git push -f' can overwrite remote history. Use with caution, especially in shared repositories.
    git push -f || { echo "ERROR: 'git push -f' failed." >&2; exit 1; }
    echo "INFO: Force push successful."
fi

# rm .gitignore
# mv .gitignore.bak .gitignore

echo "INFO: Backup script completed."

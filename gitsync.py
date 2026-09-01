"""
Push/pull this project to GitHub using dulwich (pure Python - no git.exe needed).

Setup:
    Put your GitHub personal access token in a file called .git_token.txt
    in this same folder. Never paste the token into chat or commit that file.

Usage:
    python gitsync.py push
    python gitsync.py pull
"""
import sys
import os
import glob

from dulwich import porcelain
from dulwich.repo import Repo

REPO_URL = "https://github.com/Josh-Prints/ai-race-game.git"
TOKEN_FILE = ".git_token.txt"
TRACKED_EXTENSIONS = ["*.py", "*.md"]   # add more patterns here if you want other files tracked


def get_authed_url():
    if not os.path.exists(TOKEN_FILE):
        print(f"Missing {TOKEN_FILE} - put your GitHub token in that file first.")
        sys.exit(1)
    with open(TOKEN_FILE) as f:
        token = f.read().strip()
    if not token:
        print(f"{TOKEN_FILE} is empty.")
        sys.exit(1)
    return REPO_URL.replace("https://", f"https://{token}@")


def get_or_init_repo():
    if os.path.exists(".git"):
        return Repo(".")
    print("Initializing new repo...")
    repo = Repo.init(".")
    repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/main")
    return repo


def push():
    repo = get_or_init_repo()
    url = get_authed_url()

    files = []
    for pattern in TRACKED_EXTENSIONS:
        files.extend(glob.glob(pattern))
    files = [f for f in files if f != TOKEN_FILE]  # safety: never track the token file

    if not files:
        print("No files matched to track. Nothing to push.")
        return

    print(f"Tracking: {files}")
    porcelain.add(repo=".", paths=files)

    status = porcelain.status(repo)
    if not status.staged["add"] and not status.staged["modify"] and not status.staged["delete"]:
        print("No changes to commit.")
    else:
        msg = input("Commit message: ") or "update"
        porcelain.commit(
            repo=".",
            message=msg.encode(),
            author=b"Josh <josh@example.com>",
            committer=b"Josh <josh@example.com>",
        )
        print("Committed.")

    print("Pushing to GitHub...")
    porcelain.push(repo, remote_location=url, refspecs=[b"refs/heads/main"])
    print("Done.")


def pull():
    if not os.path.exists(".git"):
        print("No local repo yet - nothing to pull into. Run push first, or clone manually.")
        return
    repo = Repo(".")
    url = get_authed_url()
    print("Pulling from GitHub...")

    # dulwich's porcelain.pull() can fail with WorkingTreeModifiedError if a tracked
    # file looks locally modified (even something harmless like a line-ending change
    # from OneDrive/an editor). Worse, when that happens it can still leave the local
    # branch ref pointed at the new commit without ever writing the new file contents
    # to disk - so a second "pull" reports success ("Done") but nothing actually
    # changed, because dulwich thinks it's already up to date.
    #
    # This tool is a one-way sync (GitHub -> this folder), so instead of trying to
    # merge local changes, always force the working tree to exactly match the
    # fetched remote branch.
    result = porcelain.fetch(repo, url)
    remote_sha = result.refs.get(b"refs/heads/main")
    if remote_sha is None:
        print("Could not find refs/heads/main on the remote.")
        return
    repo.refs[b"refs/heads/main"] = remote_sha
    porcelain.reset(repo, "hard", treeish=remote_sha)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("push", "pull"):
        print("Usage: python gitsync.py [push|pull]")
        sys.exit(1)
    if sys.argv[1] == "push":
        push()
    else:
        pull()
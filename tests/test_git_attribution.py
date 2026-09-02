"""
Commit attribution guard.

GitHub builds the repository's contributor list from commit author/committer
identity *and* from Co-Authored-By trailers. Agent sessions are instructed to
append "Co-Authored-By: Claude ..." and "Claude-Session: ..." trailers, which
is how a non-human identity appeared in this repo's contributor list once
already -- it took a history rewrite and a force-push to remove.

`.githooks/commit-msg` strips those trailers at commit time, but a hook is
local config: it is off until someone runs `git config core.hooksPath
.githooks`, and it never runs at all for a commit made with --no-verify or in
a fresh clone. This test is the backstop that catches what the hook missed,
and it runs in CI where the hook definitively is not installed.

Deliberately checks the whole reachable history rather than only HEAD, so a
bad commit cannot slip in behind a later clean one.
"""

import pathlib
import re
import shutil
import subprocess

import pytest

# Substrings that must not appear in an identity field. Matched
# case-insensitively.
BANNED_ATTRIBUTION_SUBSTRINGS = ["claude", "anthropic"]

# The trailer token is banned ANYWHERE in a message, not only at line start.
#
# These were line-anchored, on the reasoning that GitHub attributes from
# trailer lines and so a commit that *describes* the policy should stay legal.
# That reasoning is why this file exists in its current form: the commit that
# introduced .githooks/commit-msg quoted the token inside its own prose, the
# anchored patterns passed it, and the literal string went to GitHub in pushed
# history. Anchoring measured the parser instead of the risk.
#
# Write "co-author trailer" when discussing it -- this comment does. The cost
# is losing the ability to quote the token verbatim in a commit message, which
# is nothing next to re-auditing a contributor list.
#
# These mirror .githooks/pre-push; change them together. .githooks/commit-msg
# stays anchored on purpose -- it edits the message rather than refusing it,
# and an unanchored match there would mangle prose instead of blocking it.
BANNED_TRAILER_PATTERNS = [
    re.compile(r"Co-authored-by:", re.IGNORECASE),
    re.compile(r"Claude-Session:", re.IGNORECASE),
    re.compile(r"Generated with \[Claude Code\]", re.IGNORECASE),
    re.compile(r"https://claude\.ai/code/session_", re.IGNORECASE),
]

# ASCII unit/record separators: chosen as delimiters because git will never
# emit them from a name, email, or message body, so parsing cannot be confused
# by a message that happens to contain the delimiter.
_FIELD_SEP = "\x1f"
_RECORD_SEP = "\x1e"

# %H sha, %an/%ae author, %cn/%ce committer, %B raw message body.
_LOG_FORMAT = _FIELD_SEP.join(["%H", "%an", "%ae", "%cn", "%ce", "%B"]) + _RECORD_SEP


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def _git_log_records():
    """Every commit reachable from any ref, as (sha, [labelled fields])."""
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    root = _repo_root()
    if not (root / ".git").exists():
        pytest.skip("not a git checkout (source tarball or exported tree)")

    result = subprocess.run(
        ["git", "log", "--all", "--no-color", f"--format={_LOG_FORMAT}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"git log failed: {result.stderr.strip()}")

    records = []
    for raw in result.stdout.split(_RECORD_SEP):
        raw = raw.strip("\n")
        if not raw:
            continue
        sha, author_name, author_email, committer_name, committer_email, message = raw.split(
            _FIELD_SEP
        )
        records.append(
            (
                sha,
                [
                    ("author name", author_name),
                    ("author email", author_email),
                    ("committer name", committer_name),
                    ("committer email", committer_email),
                ],
                message,
            )
        )
    return records


def test_no_agent_attribution_in_commit_history():
    """No commit may name an agent as author, committer, or co-author."""
    records = _git_log_records()
    assert records, "git log returned no commits -- the guard would be vacuous"

    # Collect every offender rather than failing on the first: when a rewrite
    # is needed, the useful output is the full list of SHAs to rewrite.
    offenders = []
    for sha, identity_fields, message in records:
        for label, value in identity_fields:
            lowered = value.lower()
            hits = [b for b in BANNED_ATTRIBUTION_SUBSTRINGS if b in lowered]
            if hits:
                offenders.append(f"{sha[:12]} {label}: {hits} in {value.strip()!r}")

        for line in message.splitlines():
            if any(p.search(line) for p in BANNED_TRAILER_PATTERNS):
                offenders.append(f"{sha[:12]} message trailer: {line.strip()!r}")

    assert not offenders, "agent attribution found in commit history:\n" + "\n".join(offenders)


def test_commit_msg_hook_is_present_and_executable():
    """The hook the guard backstops must exist -- a deleted hook should fail
    here rather than quietly leaving only the after-the-fact check."""
    hook = _repo_root() / ".githooks" / "commit-msg"
    assert hook.is_file(), f"{hook} is missing"
    import os

    assert os.access(hook, os.X_OK), f"{hook} is not executable (chmod +x it)"


# --- the pre-push gate -------------------------------------------------------
#
# The tests above read this repo's own history, which is detection after the
# fact. The ones below drive .githooks/pre-push in a throwaway repo with a
# local bare remote, because the thing worth proving about a gate is that it
# actually refuses, not that the file exists.


def _run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def _commit(repo, message, env=None):
    """Commit with --no-verify, i.e. with the commit-msg stripper deliberately
    bypassed. That is the exact hole pre-push exists to cover: a message the
    stripper never got to see."""
    (repo / "f.txt").write_text(message)
    _run(["git", "add", "f.txt"], cwd=repo)
    return _run(["git", "commit", "--no-verify", "-m", message], cwd=repo)


@pytest.fixture
def pushable_repo(tmp_path):
    """A scratch repo wired to a local bare remote, with this repo's real
    .githooks installed. No network: the remote is a directory."""
    if shutil.which("git") is None:
        pytest.skip("git is not available")

    remote = tmp_path / "remote.git"
    repo = tmp_path / "work"
    remote.mkdir()
    repo.mkdir()
    _run(["git", "init", "--bare", "-b", "main"], cwd=remote)
    _run(["git", "init", "-b", "main"], cwd=repo)
    for key, value in (
        ("user.name", "Test Person"),
        ("user.email", "test@example.com"),
        # The hooks under test, by absolute path -- the scratch repo is not
        # inside this project.
        ("core.hooksPath", str(_repo_root() / ".githooks")),
    ):
        _run(["git", "config", key, value], cwd=repo)
    _run(["git", "remote", "add", "origin", str(remote)], cwd=repo)
    return repo


def test_pre_push_hook_is_present_and_executable():
    hook = _repo_root() / ".githooks" / "pre-push"
    assert hook.is_file(), f"{hook} is missing"
    import os

    assert os.access(hook, os.X_OK), f"{hook} is not executable (chmod +x it)"


def test_pre_push_allows_a_clean_commit(pushable_repo):
    """The gate must not be so blunt that ordinary work cannot leave the
    machine -- a guard that blocks everything gets disabled within a day."""
    _commit(pushable_repo, "Add a thing")
    pushed = _run(["git", "push", "origin", "main"], cwd=pushable_repo)
    assert pushed.returncode == 0, pushed.stderr


def test_pre_push_rejects_a_co_authored_by_trailer(pushable_repo):
    _commit(
        pushable_repo,
        "Add a thing\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
    )
    pushed = _run(["git", "push", "origin", "main"], cwd=pushable_repo)
    assert pushed.returncode != 0, "push succeeded; the trailer reached the remote"
    assert "refusing to push" in pushed.stderr
    assert "trailer" in pushed.stderr


def test_pre_push_rejects_an_agent_identity(pushable_repo):
    """Identity is the axis commit-msg cannot touch: it edits the message and
    has no say over author/committer. GitHub credits from identity first."""
    _run(["git", "config", "user.name", "Claude"], cwd=pushable_repo)
    _run(["git", "config", "user.email", "noreply@anthropic.com"], cwd=pushable_repo)
    _commit(pushable_repo, "Add a thing")
    pushed = _run(["git", "push", "origin", "main"], cwd=pushable_repo)
    assert pushed.returncode != 0, "push succeeded; the identity reached the remote"
    assert "identity" in pushed.stderr


def test_pre_push_rejects_the_token_even_in_prose(pushable_repo):
    """The case that cost a repository.

    The commit introducing .githooks/commit-msg quoted the trailer token inside
    a sentence explaining what the hook strips. Every anchored check passed it,
    so the literal string reached GitHub in pushed history. Prose is no longer
    an exemption."""
    _commit(
        pushable_repo,
        'Explain the guard\n\nAgent sessions append "Co-Authored-By: Someone"\n'
        "to every commit, so the hook strips it.",
    )
    pushed = _run(["git", "push", "origin", "main"], cwd=pushable_repo)
    assert pushed.returncode != 0, "prose containing the token reached the remote"
    assert "refusing to push" in pushed.stderr


def test_pre_push_allows_the_approved_phrasing(pushable_repo):
    """The rule has to leave a way to write about itself, or it gets disabled.
    Saying "co-author trailer" instead of the literal token is that way, and it
    is what the hooks and this module now do throughout."""
    _commit(
        pushable_repo,
        "Explain the guard\n\nGitHub reads an agent co-author trailer and "
        "credits the named\nidentity, so the hook strips it.",
    )
    pushed = _run(["git", "push", "origin", "main"], cwd=pushable_repo)
    assert pushed.returncode == 0, pushed.stderr

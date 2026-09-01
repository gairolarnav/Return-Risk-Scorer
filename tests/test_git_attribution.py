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

# Message bodies get line-anchored trailer patterns instead of a substring
# scan, because GitHub attributes from trailer lines, not from prose. A commit
# that *describes* this policy -- the one that added the hook does exactly
# that -- must stay legal, or the guard punishes documenting itself.
#
# These mirror the patterns in .githooks/commit-msg; change them together.
BANNED_TRAILER_PATTERNS = [
    re.compile(r"^Co-authored-by:.*(claude|anthropic)", re.IGNORECASE),
    re.compile(r"^Claude-Session:\s", re.IGNORECASE),
    re.compile(r"^.{0,4}Generated with \[Claude Code\]", re.IGNORECASE),
    re.compile(r"^\s*https://claude\.ai/code/session_", re.IGNORECASE),
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

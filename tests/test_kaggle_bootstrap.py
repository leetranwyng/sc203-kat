"""Offline Git integration tests; no GitHub login, model or GPU required."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

# During migration the test can run alongside the staged helper. In the repo it
# imports the installed/editable package instead.
staged = Path(__file__).with_name("kaggle.py")
if staged.is_file():
    spec = importlib.util.spec_from_file_location("staged_kaggle", staged)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
else:
    from sc203_kat.runtime import kaggle as bootstrap


class GitBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sc203-git-test-")
        self.root = Path(self.temp.name)
        self.env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never",
                        GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="protocol.file.allow",
                        GIT_CONFIG_VALUE_0="always")
        self.origin = self.repository("origin")
        self.first = self.commit(self.origin, "first")
        self.git(self.origin, "tag", "v1")
        self.target = self.root / "checkout"

    def tearDown(self):
        self.temp.cleanup()

    def git(self, root, *args, check=True):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, encoding="utf8", errors="replace", env=self.env)
        if check and result.returncode:
            self.fail(result.stderr)
        return result.stdout.strip() if check else result

    def repository(self, name):
        path = self.root / name
        path.mkdir()
        self.git(path, "init", "-b", "main")
        self.git(path, "config", "user.name", "SC203 fixture")
        self.git(path, "config", "user.email", "fixture@localhost")
        return path

    def commit(self, repo, value):
        (repo / "value.txt").write_text(value, encoding="utf8")
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-m", value)
        return self.git(repo, "rev-parse", "HEAD")

    def test_first_clone_is_detached_at_fetched_tag(self):
        result = bootstrap.checkout(self.origin.as_uri(), "v1", self.target, env=self.env)
        self.assertEqual(result, self.target.resolve())
        self.assertEqual(self.git(result, "rev-parse", "HEAD"), self.first)
        self.assertEqual(self.git(result, "symbolic-ref", "-q", "HEAD", check=False).returncode, 1)
        self.assertEqual((result / "value.txt").read_text(), "first")
        self.assertEqual(bootstrap.source_metadata(result, env=self.env)["commit"], self.first)

    def test_cached_main_and_new_tag_fetch_new_remote_commit(self):
        bootstrap.checkout(self.origin.as_uri(), "main", self.target, env=self.env)
        second = self.commit(self.origin, "second")
        self.git(self.origin, "tag", "v2")
        bootstrap.checkout(self.origin.as_uri(), "main", self.target, env=self.env)
        self.assertEqual(self.git(self.target, "rev-parse", "HEAD"), second)
        bootstrap.checkout(self.origin.as_uri(), "v2", self.target, env=self.env)
        self.assertEqual((self.target / "value.txt").read_text(), "second")

    def test_full_commit_pin_can_fetch_older_commit(self):
        self.commit(self.origin, "second")
        bootstrap.checkout(self.origin.as_uri(), self.first, self.target, env=self.env)
        self.assertEqual(self.git(self.target, "rev-parse", "HEAD"), self.first)
        self.assertEqual((self.target / "value.txt").read_text(), "first")

    def test_dirty_source_is_refused_without_overwriting(self):
        bootstrap.checkout(self.origin.as_uri(), "main", self.target, env=self.env)
        (self.target / "value.txt").write_text("local change", encoding="utf8")
        with self.assertRaisesRegex(RuntimeError, "dirty checkout"):
            bootstrap.checkout(self.origin.as_uri(), "main", self.target, env=self.env)
        self.assertEqual((self.target / "value.txt").read_text(), "local change")

    def test_origin_mismatch_is_refused(self):
        bootstrap.checkout(self.origin.as_uri(), "main", self.target, env=self.env)
        other = self.repository("other")
        self.commit(other, "other")
        with self.assertRaisesRegex(ValueError, "origin differs"):
            bootstrap.checkout(other.as_uri(), "main", self.target, env=self.env)
        self.assertEqual(self.git(self.target, "rev-parse", "HEAD"), self.first)

    def test_existing_non_git_files_are_refused(self):
        self.target.mkdir()
        (self.target / "keep.txt").write_text("keep", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "already contains files"):
            bootstrap.checkout(self.origin.as_uri(), "main", self.target, env=self.env)
        self.assertEqual((self.target / "keep.txt").read_text(), "keep")

    def submodule_fixture(self):
        one, two = self.repository("one"), self.repository("two")
        pin = self.commit(one, "module one")
        self.commit(two, "module two")
        self.git(self.origin, "submodule", "add", "--name", "apc_vlm", one.as_uri(), "third_party/apc_vlm")
        self.git(self.origin, "submodule", "add", "--name", "pcd", two.as_uri(), "third_party/pcd")
        self.git(self.origin, "add", ".")
        self.git(self.origin, "commit", "-m", "add pinned paper sources")
        return one, pin

    def test_only_selected_submodule_is_initialized_at_gitlink(self):
        one, pin = self.submodule_fixture()
        self.commit(one, "new upstream commit")
        bootstrap.checkout(self.origin.as_uri(), "main", self.target, method="apc_vis", env=self.env)
        selected = self.target / "third_party/apc_vlm"
        other = self.target / "third_party/pcd"
        self.assertTrue((selected / ".git").is_file())
        self.assertFalse((other / ".git").exists())
        self.assertEqual(self.git(selected, "rev-parse", "HEAD"), pin)
        metadata = bootstrap.source_metadata(self.target, "apc_vis", env=self.env)
        self.assertEqual(metadata["submodule"]["commit"], pin)
        self.assertEqual(metadata["submodule"]["pinned_commit"], pin)

    def test_dirty_submodule_is_refused(self):
        self.submodule_fixture()
        bootstrap.checkout(self.origin.as_uri(), "main", self.target, method="apc_vis", env=self.env)
        selected = self.target / "third_party/apc_vlm"
        (selected / "value.txt").write_text("local module change", encoding="utf8")
        with self.assertRaisesRegex(RuntimeError, "dirty checkout"):
            bootstrap.ensure_submodule(self.target, "apc_vis", env=self.env)
        self.assertEqual((selected / "value.txt").read_text(), "local module change")

    def test_option_like_ref_is_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap.checkout(self.origin.as_uri(), "--upload-pack=bad", self.target, env=self.env)


if __name__ == "__main__":
    unittest.main()

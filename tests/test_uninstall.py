import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from icommand.cli import cli
from icommand.embeddings import get_local_model_cache_dir


class UninstallTests(unittest.TestCase):
    def test_get_local_model_cache_dir_uses_current_hf_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "hf-cache"
            repo_dir = cache_root / "models--Snowflake--snowflake-arctic-embed-xs"
            (repo_dir / "snapshots" / "commit123").mkdir(parents=True)

            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(cache_root)}, clear=False):
                self.assertEqual(get_local_model_cache_dir(), repo_dir)

    def test_uninstall_removes_local_model_cache_and_preserves_other_repos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            home = root / "home"
            home.mkdir()
            hook_rc = home / ".zshrc"
            hook_rc.write_text(
                '# icommand: AI-powered command history search\nsource "/tmp/hook.sh"\n'
            )

            icommand_dir = home / ".icommand"
            icommand_dir.mkdir()
            (icommand_dir / "history.db").write_text("db")

            hf_cache = root / "hf-cache"
            target_repo = hf_cache / "models--Snowflake--snowflake-arctic-embed-xs"
            (target_repo / "snapshots" / "commit123").mkdir(parents=True)
            (target_repo / "snapshots" / "commit123" / "tokenizer.json").write_text("{}")

            other_repo = hf_cache / "models--someone-else--different-model"
            (other_repo / "snapshots" / "commit999").mkdir(parents=True)
            (other_repo / "snapshots" / "commit999" / "config.json").write_text("{}")

            runner = CliRunner()
            with (
                mock.patch("pathlib.Path.home", return_value=home),
                mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hf_cache)}, clear=False),
                mock.patch("icommand.cli.shutil.which", return_value=None),
                mock.patch("subprocess.Popen") as popen,
            ):
                result = runner.invoke(cli, ["uninstall"], input="y\n")

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertFalse(icommand_dir.exists())
            self.assertFalse(target_repo.exists())
            self.assertTrue(other_repo.exists())
            self.assertIn("model    removed", result.output)
            popen.assert_called_once()


if __name__ == "__main__":
    unittest.main()

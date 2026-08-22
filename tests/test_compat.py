import sys
import unittest

import finrag.compat  # noqa: F401


class CompatTest(unittest.TestCase):
    def test_vertexai_stub_injected(self):
        self.assertIn("langchain_community.chat_models.vertexai", sys.modules)
        module = sys.modules["langchain_community.chat_models.vertexai"]
        self.assertTrue(hasattr(module, "ChatVertexAI"))


if __name__ == "__main__":
    unittest.main()

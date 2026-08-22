import unittest

from finrag.chunking import split_documents


class ChunkingTest(unittest.TestCase):
    def test_passage_dataset_aware(self):
        corpus = [
            {"_id": "a", "title": "T", "text": "one. " * 500},
            {"_id": "b", "title": "", "text": "two. " * 500},
        ]
        result = split_documents(corpus, chunk_size=200, chunk_overlap=20, dataset_type="passage")
        self.assertGreater(len(result.texts), 4)
        self.assertEqual(len(result.texts), len(result.ids))
        self.assertEqual(len(result.texts), len(result.original_ids))

    def test_tabular_separator_pipe(self):
        corpus = [{"_id": "a", "title": "", "text": "A|B|C|D|E|F|G|H" * 200}]
        result = split_documents(corpus, chunk_size=100, chunk_overlap=0, dataset_type="tabular", strategy="dataset-aware")
        self.assertGreater(len(result.texts), 1)
        # Pipe-based chunks should keep row-like fragments (no guarantee each chunk has pipe)
        self.assertTrue(all(t.strip() for t in result.texts))

    def test_semantic_strategy(self):
        corpus = [{"_id": "a", "title": "", "text": "First sentence. Second sentence. Third sentence."}]
        result = split_documents(corpus, chunk_size=30, chunk_overlap=0, strategy="semantic")
        self.assertGreaterEqual(len(result.texts), 2)


if __name__ == "__main__":
    unittest.main()

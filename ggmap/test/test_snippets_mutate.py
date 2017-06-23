from unittest import TestCase, main
from random import seed

from ggmap.snippets import (mutate_sequence)


class SnippetTests(TestCase):
    def setUp(self):
        seed(42)
        self.sequence = 'acgtacgtacgt'

    def test_mutate_sequence_default(self):
        mut_seq = mutate_sequence(self.sequence)
        self.assertEqual(mut_seq, "acgtacgtacAt")

    def test_mutate_sequence_five(self):
        mut_seq = mutate_sequence(self.sequence, num_mutations=5)
        self.assertEqual(mut_seq, 'TcTtCAgtacAt')

    def test_mutate_sequence_toomany(self):
        with self.assertRaisesRegex(ValueError,
                                    ("Sequence not long enough for "
                                     "that many mutations.")):
            mutate_sequence(self.sequence,
                            num_mutations=len(self.sequence)+1)

    def test_mutate_sequence_diffAlphabet(self):
        mut_seq = mutate_sequence(self.sequence, num_mutations=4,
                                  alphabet=set(['#']))
        self.assertEqual(mut_seq, '#cg##cgtac#t')

    def test_mutate_sequence_tooSmallAlphabet(self):
        with self.assertRaisesRegex(ValueError,
                                    "Alphabet is too small to find mutation!"):
            for i in range(1, 10):
                mutate_sequence(self.sequence, num_mutations=1,
                                alphabet=set(['a']))


if __name__ == '__main__':
    main()

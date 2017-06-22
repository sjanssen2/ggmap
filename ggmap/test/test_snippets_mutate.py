from unittest import TestCase, main
from random import seed

from ggmap.snippets import (mutate_sequence)


class SnippetTests(TestCase):
    def setUp(self):
        seed(42)
        self.sequence = 'acgtacgtacgt'

    def test_mutate_sequence_default(self):
        mut_seq = mutate_sequence(self.sequence)
        self.assertEqual(sum(map(lambda x: x.isupper(), mut_seq)), 1)

    def test_mutate_sequence_five(self):
        mut_seq = mutate_sequence(self.sequence, num_mutations=5)
        self.assertEqual(sum(map(lambda x: x.isupper(), mut_seq)), 5)

    def test_mutate_sequence_toomany(self):
        with self.assertRaisesRegex(ValueError,
                                    ("Sequence not long enough for "
                                     "that many mutations.")):
            mutate_sequence(self.sequence,
                            num_mutations=len(self.sequence)+1)

    def test_mutate_sequence_diffAlphabet(self):
        mut_seq = mutate_sequence(self.sequence, num_mutations=4,
                                  alphabet=set(['#']))
        self.assertEqual(sum(map(lambda x: x == '#', mut_seq)), 4)


if __name__ == '__main__':
    main()

from unittest import TestCase, main

from ggmap.snippets import (_time_torque2slurm)


class SnippetTests(TestCase):
    def test__time_torque2slurm(self):
        self.assertEqual(_time_torque2slurm("48:76:88"), '2-1:17')
        self.assertEqual(_time_torque2slurm("48:06:88"), '2-0:7')
        self.assertEqual(_time_torque2slurm("00:00:08"), '0-0:1')
        self.assertEqual(_time_torque2slurm("04:00:00"), '0-4:0')


if __name__ == '__main__':
    main()

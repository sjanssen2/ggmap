from unittest import TestCase, main

from ggmap.analyses import _executor


class ExecutorTests(TestCase):
    def test__executor(self):
        def pre_execute(workdir, args):
            file_mapping = workdir + '/headermap.tsv'
            m = open(file_mapping, 'w')
            m.write('pre_execute test\n')
            m.close()

        def commands(workdir, ppn, args):
            commands = ['python --version >> %s 2>&1' %
                        (workdir+'/headermap.tsv')]
            return commands

        def post_execute(workdir, args, pre_data):
            cont = []
            f = open(workdir+'/headermap.tsv', 'r')
            for line in f.readlines():
                cont.append(line)
            f.close()
            return cont

        res = _executor('travistest',
                        {'seqs': 'fake'},
                        pre_execute,
                        commands,
                        post_execute,
                        dry=False, wait=True, use_grid=False, nocache=True,
                        timing=False)
        self.assertIn('Python 2.7.11 :: Continuum Analytics, Inc.',
                      res['results'][1])
        self.assertNotIn('Python 3.', res['results'][1])


if __name__ == '__main__':
    main()

# This file shall contain all code that is necessary to execute other software
# either locally or via a SLURM grid. It contains mechanisms for caching, ...

from sys import stderr, stdout
import os
from hashlib import md5, sha256
from collections import OrderedDict
from tempfile import gettempdir, mkdtemp
import subprocess
from time import sleep, time
from datetime import datetime
from pickle import dump, load

import pandas as pd

from skbio.stats.distance import DistanceMatrix
from biom.table import Table

from ggmap import settings
settings.init()

def cache(func):
    """Decorator: Cache results of a function call to disk. It's the users
       obligation to assign the cached data to the according function through
       the choice of a filename for the cache. It does NOT depend on the
       function arguments.

    Parameters
    ----------
    func : executabale
        A function plus parameters whichs results shall be cached, e.g.
        "fct_example(1,5,3)", where
        @cache
        def fct_test(a, b, c):
            return a + b * c
    cache_filename : str
        Default: None. I.e. caching is deactivated.
        Pathname to cache file, which will hold results of the function call.
        If file exists, results are loaded from it instead of recomputing via
        provided function. Otherwise, function will be executed and results
        stored to this file.
    cache_verbose : bool
        Default: True.
        Report caching status to 'cache_err', which by default is sys.stderr.
    cache_err : StringIO
        Default: sys.stderr.
        Stream onto which status messages shall be printed.
    cache_force_renew : bool
        Default: False.
        Force re-execution of provided function even if cache file exists.

    Returns
    -------
    Results of provided function, either by actually executing the function
    with provided parameters or by loaded results from filename.

    Notes
    -----
    It is the obligation of the user to ensure that arguments for the provided
    function don't change between creation of cache file and loading from cache
    file!
    """
    func_name = func.__name__

    def execute(*args, **kwargs):
        cache_args = {'cache_filename': None,
                      'cache_verbose': True,
                      'cache_err': stderr,
                      'cache_force_renew': False}
        for varname in cache_args.keys():
            if varname in kwargs:
                cache_args[varname] = kwargs[varname]
                del kwargs[varname]

        if cache_args['cache_filename'] is None:
            if cache_args['cache_verbose']:
                cache_args['cache_err'].write(
                    '%s: no caching, since "cache_filename" is None.\n' %
                    func_name)
            return func(*args, **kwargs)

        if os.path.exists(cache_args['cache_filename']) and\
           (os.stat(cache_args['cache_filename']).st_size <= 0):
            if cache_args['cache_verbose']:
                cache_args['cache_err'].write(
                    '%s: removed empty cache.\n' %
                    func_name)
            os.remove(cache_args['cache_filename'])

        if (not os.path.exists(cache_args['cache_filename'])) or\
           cache_args['cache_force_renew']:
            try:
                f = open(cache_args['cache_filename'], 'wb')
                results = func(*args, **kwargs)
                dump(results, f)
                f.close()
                if cache_args['cache_verbose']:
                    cache_args['cache_err'].write(
                        '%s: stored results in cache "%s".\n' %
                        (func_name, cache_args['cache_filename']))
            except Exception as e:
                raise e
        else:
            f = open(cache_args['cache_filename'], 'rb')
            results = load(f)
            f.close()
            if cache_args['cache_verbose']:
                cache_args['cache_err'].write(
                    '%s: retrieved results from cache "%s".\n' %
                    (func_name, cache_args['cache_filename']))
        return results
    if func.__doc__ is not None:
        execute.__doc__ = func.__doc__
    else:
        execute.__doc__ = ""
    execute.__doc__ += "\n\n" + cache.__doc__
    # restore wrapped function name
    execute.__name__ = func_name
    return execute


def _time_torque2slurm(t_time):
    """Convertes run-time resource string from Torque to Slurm.
    Input format is hh:mm:ss, output is <days>-<hours>:<minutes>

    Parameters
    ----------
    t_time : str
        Input time duration in format hh:mm:ss

    Returns
    -------
    Slurm compatible time duration.
    """
    t_hours, t_minutes, t_seconds = map(int, t_time.split(':'))
    s_minutes = (t_seconds // 60) + t_minutes
    s_hours = (s_minutes // 60) + t_hours
    s_minutes = s_minutes % 60
    s_days = s_hours // 24
    s_hours = s_hours % 24

    # set a minimal run time, if Torque time is < 60 seconds
    if (s_days == 0) and (s_hours == 0) and (s_minutes == 0):
        s_minutes = 1

    return "%i-%02i:%02i:00" % (s_days, s_hours, s_minutes)


def _add_timing_cmds(commands, file_timing):
    """Change list of commands, such that system's time is used to trace
       run-time.

    Parameters
    ----------
    commands : [str]
        List of commands.
    file_timing : str
        Filepath to the file into which timing information shall be written

    Returns
    -------
    [str] list of changed commands with timing capability.
    """
    timing_cmds = []
    # report machine name
    timing_cmds.append('uname -a > %s' % file_timing)
    # report commands to be executed (I have problems with quotes)
    # timing_cmds.append('echo `%s` >> ${PBS_JOBNAME}.t${PBS_JOBID}'
    #                    % '; '.join(cmds))
    # add time to every command
    for cmd in commands:
        # cd cannot be timed and any attempt will fail changing the
        # directory
        if cmd.startswith('cd ') or\
           cmd.startswith('module load ') or\
           cmd.startswith('var_') or\
           cmd.startswith('export ') or\
           cmd.startswith('source ') or\
           cmd.startswith('ulimit '):
                timing_cmds.append(cmd)
        elif cmd.startswith('if [ '):
            ifcon, rest = re.findall(
                r'(if \[.+?\];\s*then\s*)(.+)', cmd, re.IGNORECASE)[0]
            timing_cmds.append(('%s '
                                '%s '
                                '-v '
                                '-o %s '
                                '-a %s') %
                               (ifcon, settings.EXEC_TIME, file_timing, rest))
        else:
            timing_cmds.append(('%s '
                                '-v '
                                '-o %s '
                                '-a %s') %
                               (settings.EXEC_TIME, file_timing, cmd))
    return timing_cmds


def get_conda_activate_cmd(use_grid, environment):
    if settings.GRIDNAME == 'JLU':
        # but remember to to create the ~/.bash_profile file and copy and paste conda init script from .bashrc!
        if use_grid is False:
            cmd_conda = "source %s/etc/profile.d/conda.sh; conda activate %s; " % (settings.DIR_CONDA, environment)
        else:
            cmd_conda = "conda activate %s; " % (environment)
    elif settings.GRIDNAME == 'JLU_SLURM':
        cmd_conda = "source %s/etc/profile.d/conda.sh; conda activate %s; " % (settings.DIR_CONDA, environment)
    else:
        cmd_conda = "source %s/etc/profile.d/conda.sh; %s/condabin/conda activate %s; " % (
            settings.DIR_CONDA, settings.DIR_CONDA, environment)
    return cmd_conda


def cluster_run(cmds, jobname, result, environment=None,
                walltime='4:00:00', nodes=1, ppn=10, pmem='8GB',
                gebin=settings.GRIDENGINE_BINDIR, dry=True, wait=False,
                file_qid=None, file_condaenvinfo=None, out=stdout,
                err=stderr, timing=False, file_timing=None, array=1,
                use_grid=settings.USE_GRID,
                force_slurm=False, no_mail=False):
    """ Submits a job to the cluster.

    Paramaters
    ----------
    cmds : [str]
        List of commands to be run on the cluster.
    jobname : str
        A name for the cluster job.
    result : path
        A file or dir holding results of a sucessful run. Don't re-submit if
        result exists.
    environment : str
        Name of a conda environment to activate.
    walltime : str
        Format hh:mm:ss maximal CPU time for the job. Default: '4:00:00'.
    nodes : int
        Number of nodes onto the job should be distributed. Defaul: 1
    ppn : int
        Number of cores within one node onto which the job should be
        distributed. Default 10.
    pmem : str
        Format 'xGB'. Memory requirement per ppn for the job, e.g. if ppn=10
        and pmem=8GB the node must have at least 80GB free memory.
        Default: '8GB'.
    gebin : path
        Path to the dir holding SGE binaries.
        Default: /opt/torque-4.2.8/bin
    dry : bool
        Only print command instead of executing it. Good for debugging.
        Default = True
    wait : bool
        Wait for job completion before qsub's return
    file_qid : str
        Default None. Create a file containing the qid of the submitted job.
        This will ease identification of TMP working directories.
    file_condaenvinfo : str
        Default: None.
        If specified, AND environment is not None,
        the result of "conda list --name X" is written to this file.
    out : StringIO
        Buffer onto which messages should be printed. Default is sys.stdout.
    err : StringIO
        Default: sys.stderr.
        Buffer for status reports.
    timing : bool
        If True than add time output to every command and store in cr_*.t*
        file. Default is False.
    file_timing : str
        Default: None
        Define filepath into which timeing information shall be written.
    array : int
        Default: 1
        If > 1 than an array job is submitted. Make sure in- and outputs can
        deal with ${PBS_ARRAY_INDEX}!
        Only available for Torque.
    use_grid : bool
        Defaul: True.
        If False, commands are executed locally instead of submitting them to
        a HPC (= either Torque or Slurm).
    force_slurm : bool
        Default: False.
        If True, cluster_run is enforeced to choose slurm instead of auto
        detection based on machine node name.
    no_mail : bool
        Default: False
        If True, will not send emails about exit status when complete.

    Returns
    -------
    Cluster job ID as str.
    """
    VALID_CMDS_KEYS = ['pre', 'main', 'post']

    if result is None:
        raise ValueError("You need to specify a result path.")
    parent_res_dir = "/".join(result.split('/')[:-1])
    if not os.access(parent_res_dir, os.W_OK):
        raise ValueError("Parent result directory '%s' is not writable!" %
                         parent_res_dir)
    if file_qid is not None:
        if not os.access('/'.join(file_qid.split('/')[:-1]), os.W_OK):
            raise ValueError("Cannot write qid file '%s'." % file_qid)
    if os.path.exists(result):
        if err:
            err.write("%s already computed\n" % jobname)
        return "Result already present!"
    if jobname is None:
        raise ValueError("You need to set a jobname!")
    if len(jobname) <= 1:
        raise ValueError("You need to set non empty jobname!")

    if isinstance(cmds, str):
        cmds = [cmds]
    # new mechanism: I want to enable job dependencies, i.e. some commands need
    # to finish (e.g. prepare command input files) before an array job can
    # highly parallel be executed e.g. rarefaction iterations.
    # Therefore, I expect cmds to be a dictionary with keys 'pre' 'main' and
    # 'post'
    if isinstance(cmds, dict):
        if set(cmds.keys() - set(VALID_CMDS_KEYS)) != set([]):
            raise ValueError(
                ("your command dictionary has unknown keys: '%s'. "
                 "Please only use 'pre', 'main', and 'post'!") % "','".join(
                    set(cmds.keys() - set(VALID_CMDS_KEYS))))
    elif isinstance(cmds, list):
        # no specific command category given, assume all commands shall be "main"
        cmds = {'pre': [], 'main': cmds, 'post': []}
    assert isinstance(cmds, dict)

    for cmdtype in VALID_CMDS_KEYS:
        for cmd in cmds[cmdtype]:
            if "'" in cmd:
                raise ValueError("One of your commands contain a ' char. "
                                 "Please remove!")

    fps_timing = {k: None for k in VALID_CMDS_KEYS}
    if timing:
        for cmdtype in VALID_CMDS_KEYS:
            if file_timing is None:
                fps_timing[cmdtype] = '%s.t${%s}.%s' % (jobname, settings.VARNAME_PBSARRAY, cmdtype)
            else:
                fps_timing[cmdtype] = '%s.%s' % (file_timing, cmdtype)
            if cmdtype != 'main':
                fps_timing[cmdtype] = fps_timing[cmdtype].replace('${%s}' % settings.VARNAME_PBSARRAY, '')
            cmds[cmdtype] = _add_timing_cmds(cmds[cmdtype], fps_timing[cmdtype])

    cmd_list = {k: "" for k in VALID_CMDS_KEYS}
    cmd_conda = ""
    env_present = None
    fps_scripts = dict()
    if environment is not None:
        if file_condaenvinfo is None:
            file_condaenvinfo = ""
        else:
            file_condaenvinfo = " > %s" % file_condaenvinfo
        # check if environment exists
        if '/' not in environment:  # special case where we use environments in non standard paths
            with subprocess.Popen("%s/condabin/conda list -n %s %s" % (settings.DIR_CONDA, environment, file_condaenvinfo),
                                  shell=True,
                                  stdout=subprocess.PIPE) as env_present:
                if (env_present.wait() != 0):
                    raise ValueError("Conda environment '%s' not present." %
                                     environment)
        cmd_conda = get_conda_activate_cmd(use_grid, environment)
        # if settings.GRIDNAME == 'JLU':
        #     # but remember to to create the ~/.bash_profile file and copy and paste conda init script from .bashrc!
        #     if use_grid is False:
        #         cmd_conda = "source %s/etc/profile.d/conda.sh; conda activate %s; " % (settings.DIR_CONDA, environment)
        #     else:
        #         cmd_conda = "conda activate %s; " % (environment)
        # elif settings.GRIDNAME == 'JLU_SLURM':
        #     cmd_conda = "source %s/etc/profile.d/conda.sh; conda activate %s; " % (settings.DIR_CONDA, environment)
        # else:
        #     cmd_conda = "source %s/etc/profile.d/conda.sh; %s/condabin/conda activate %s; " % (
        #         settings.DIR_CONDA, settings.DIR_CONDA, environment)

    slurm = False
    if use_grid is False:
        cmd_list['SPAWN'] = cmd_conda
        cmd_list['SPAWN'] += " && ".join(cmds['pre'])
        if len(cmds['pre']) > 0:
            cmd_list['SPAWN'] += ' && '
        cmd_list['SPAWN'] += ' for %s in `seq 1 %i`; do %s; done;' % (
            settings.VARNAME_PBSARRAY, array, " && ".join(cmds['main']))
        cmd_list['SPAWN'] += ' %s' % " && ".join(cmds['post'])
    else:
        pwd = subprocess.check_output(["pwd"]).decode('ascii').rstrip()

        if (settings.GRIDNAME == 'USF') or (settings.PREFER_SLURM):
            slurm = True
        else:
            slurm = False
        with subprocess.Popen("which srun" if slurm else "which qsub",
                              shell=True, stdout=subprocess.PIPE,
                              executable="bash") as call_x:
            if call_x.wait() != 0:
                msg = ("You don't seem to have access to a grid!")
                if dry:
                    if err is not None:
                        err.write(msg)
                else:
                    raise ValueError(msg)
        if force_slurm:
            slurm = True

        if slurm is False:
            highmem = ''
            if settings.GRIDNAME == 'barnacle':
                if ppn * int(pmem[:-2]) > 250:
                    highmem = ':highmem'
            files_loc = ''
            if file_qid is not None:
                files_loc = ' -o %s/ -e %s/ ' % tuple(
                    ["/".join(file_qid.split('/')[:-1])] * 2)

            flag_array = ''
            if array > 1:
                if settings.GRIDNAME == 'barnacle' or settings.GRIDNAME == 'JLU':
                    flag_array = '-t 1-%i' % array
                elif settings.GRIDNAME == 'HPCHHU':
                    flag_array = '-J 1-%i' % array
            resources = " -l walltime=%s,nodes=%i%s:ppn=%i,mem=%s " % (
                walltime, nodes, highmem, ppn, pmem)
            if settings.GRIDNAME == 'JLU':
                # further differentiate between old and new 18.04 cluster (08.04.2020)
                arg_multislot = " -pe multislot %i " % ppn
                #if settings.GRIDENGINE_BINDIR == '/usr/bin/':
                    # according to Burkhard, the "new cluster" doesn't have multislots yet
                    # UPDATE: 2021-01-06: "das PE ist da, sollte auch funktionieren"
                #    arg_multislot = ""
                pmem_value = pmem
                if pmem is None:
                    pmem_value = '8GB'
                else:
                    pmem_value = pmem[:-1] if pmem.upper().endswith('B') else pmem
                resources = " -l virtual_free=%s %s -S /bin/bash " % (pmem_value, arg_multislot)
            ge_cmd = (
                ("%s/qsub %s %s -V %s -N cr_%s %s %s -r y") %
                (gebin,
                 '-A %s' % settings.GRID_ACCOUNT if settings.GRID_ACCOUNT != "" else "",
                 "-d '%s'" % pwd if settings.GRIDNAME == 'barnqacle' else '',
                 resources,
                 jobname, flag_array, files_loc))
            cmd_list['main'] += "echo '%s%s' | %s" % (cmd_conda, " && ".join(cmds), ge_cmd)
        else:
            for cmdtype in VALID_CMDS_KEYS:
                slurm_script = "#!/bin/bash\n\n"
                slurm_script += '#SBATCH --job-name=cr_%s_%s\n' % (jobname, cmdtype)
                slurm_script += '#SBATCH --output=%s/slurmlog-%%x-%%A.%%a_%s.log\n' % (pwd if file_qid is None else os.path.abspath(os.path.dirname(file_qid)), cmdtype)
                slurm_script += '#SBATCH --error=%s/slurmlog-%%x-%%A.%%a_%s.err\n' % (pwd if file_qid is None else os.path.abspath(os.path.dirname(file_qid)), cmdtype)
                slurm_script += '#SBATCH --partition=%s\n' % settings.GRID_ACCOUNT
                slurm_script += '#SBATCH --ntasks=1\n'
                slurm_script += '#SBATCH --cpus-per-task=%i\n' % ppn
                slurm_script += '#SBATCH --mem-per-cpu=%s\n' % (pmem.upper() if pmem is not None else '8GB')
                slurm_script += '#SBATCH --time=%s\n' % _time_torque2slurm(
                    walltime)
                if cmdtype == 'main':
                    slurm_script += '#SBATCH --array=1-%i\n' % array
                if cmdtype == 'post' and (no_mail is False):
                    slurm_script += '#SBATCH --mail-type=END,FAIL\n'
                    slurm_script += '#SBATCH --mail-user=%s\n\n' % settings.GRID_EMAIL_NOTIFICATION
                slurm_script += '$(which uname) -a\n'

                for cmd in cmds[cmdtype]:
                    if cmdtype != 'main':
                        assert settings.VARNAME_PBSARRAY not in cmd, "array job can only be used in 'main'"
                    slurm_script += '%s\n' % (cmd.replace(
                        '${%s}' % settings.VARNAME_PBSARRAY, '${SLURM_ARRAY_TASK_ID}'))
                if file_qid is not None:
                    file_script = os.path.dirname(file_qid) + '/slurm_script_%s.sh' % cmdtype
                else:
                    _, file_script = mkstemp(suffix='.slurm.sh')
                fps_scripts[cmdtype] = file_script
                f = open(fps_scripts[cmdtype], 'w')
                f.write(slurm_script)
                f.close()
            # if on jupyterlab from BCF@JLU, some slurm vars are predefined for the
            # spawner process of the jupyterlab. We need to unset this specific
            # variable to avoid slurm complaining about other resource requests.
            if settings.GRIDNAME == 'JLU_SLURM':
                cmd_list['SPAWN'] = 'unset SLURM_MEM_PER_NODE && '
            cmd_list['SPAWN'] += cmd_conda
            cmd_list['SPAWN'] += ' qid_pre=`%ssbatch --parsable %s`' % (settings.GRIDENGINE_BINDIR, fps_scripts['pre'])
            cmd_list['SPAWN'] += ' && qid_main=`%ssbatch --parsable --dependency=aftercorr:$qid_pre %s`' % (settings.GRIDENGINE_BINDIR, fps_scripts['main'])
            cmd_list['SPAWN'] += ' && %ssbatch --parsable --depend=afterany:$qid_main %s' % (settings.GRIDENGINE_BINDIR, fps_scripts['post'])

    if dry is True:
        if use_grid and slurm:
            for cmdtype in VALID_CMDS_KEYS:
                out.write('CONTENT OF %s:\n' % fps_scripts[cmdtype])
                with open(fps_scripts[cmdtype], 'r') as f:
                    out.write(''.join(f.readlines()) + "\n\n")
        out.write(cmd_list['SPAWN'] + "\n")
        return None
    else:
        if use_grid is True:
            with subprocess.Popen(
                    cmd_list['SPAWN'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash') as task_qsub:
                err_msg = task_qsub.stderr.read()
                if err_msg != b"":
                    raise ValueError("Error in submitting job via qsub:\n%s" % err_msg.decode('ascii'))
                qid = task_qsub.stdout.read().decode('ascii').rstrip()
                #if settings.GRIDNAME == 'JLU':
                #    qid = qid.split(" ")[2]
                if slurm:
                    qid = qid.split()[-1]
                    if file_qid is not None:
                        os.remove(file_script)
                if file_qid is not None:
                    f = open(file_qid, 'w')
                    f.write('Cluster job ID is:\n%s\n' % qid)
                    f.close()
                job_ever_seen = False
                if wait:
                    err.write(
                        "\nWaiting for %s-cluster job %s to complete: " % ('slurm' if slurm else 'sge', qid))
                    while True:
                        if slurm:
                            with subprocess.Popen(
                                    ['squeue', '--job', qid],
                                    stdout=subprocess.PIPE) as task_squeue:
                                with subprocess.Popen(
                                        ['wc', '-l'], stdin=task_squeue.stdout,
                                        stdout=subprocess.PIPE) as task_wc:
                                    poll_status = \
                                        int(task_wc.stdout.read().decode(
                                            'ascii').rstrip())
                            # Two ore more if polling gives a table with header
                            # and one status line, i.e. job is still on the
                            # grid. Translate that to 0 of Torque.
                            # If table has only one line, i.e. the header, job
                            # terminated (hopefully successful), translate that
                            # to 1 of Torque
                            if poll_status >= 2:
                                poll_status = 0
                            else:
                                poll_status = 1
                        else:
                            poll_stati = []
                            for i in range(array):
                                p = subprocess.call(
                                    "%s/qstat %s %s" %
                                    (gebin,
                                     ' -j ' if settings.GRIDNAME == 'JLU' else '',
                                     qid.replace('[]', '[%i]' % (i+1))),
                                    shell=True)
                                poll_stati.append(p == 0)
                            if any(poll_stati):
                                poll_status = 0
                            else:
                                poll_status = 127  # some number != 0
                        if (poll_status != 0) and job_ever_seen:
                            err.write(' finished.')
                            break
                        elif (poll_status == 0) and (not job_ever_seen):
                            job_ever_seen = True
                        err.write('.')
                        sleep(10)
                else:
                    err.write("Now wait until %s job finishes.\n" % qid)
                return qid
        else:
            #if settings.GRIDNAME == 'JLU':
            #    cmd_list = 'source ~/.profile && ' + cmd_list
            with subprocess.Popen(cmd_list['SPAWN'],
                                  shell=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  executable="bash") as call_x:
                if (call_x.wait() != 0):
                    out, err = call_x.communicate()
                    raise ValueError((
                        "SYSTEM CALL FAILED.\n==== STDERR ====\n%s"
                        "\n\n==== STDOUT ====\n%s\n") % (
                            err.decode("utf-8", 'backslashreplace'),
                            out.decode("utf-8", 'backslashreplace')))
                return call_x.pid


def _parse_timing(workdir, jobname):
    """If existant, parses timing information.

    Parameters
    ----------
    workdir : str
        Path to tmp workdir of _executor containing cr_ana_<jobname>.t* file
    jobname : str
        Name of ran job.

    Parameters
    ----------
    None if file could not be found. Otherwise: [str]
    """
    files_timing = [workdir + '/' + d
                    for d in next(os.walk(workdir))[2]
                    if 'cr_ana_%s.t' % jobname in d]
    for file_timing in files_timing:
        with open(file_timing, 'r') as content_file:
            return content_file.readlines()
        # stop after reading first found file, since there should only be one
        break
    return None


def _md5(filepath):
    """Returns md5sum of file path"""
    hash_md5 = md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def pandas_hash(df):
    # sort indices (only one for Series)
    df_cache = df.sort_index()
    if isinstance(df, pd.DataFrame):
        df_cache = df_cache.sort_index(axis=1)

    # generate one hash per row with pandas own lib
    row_hashes = pd.util.hash_pandas_object(df_cache, index=True)

    # combine all row-hashes into one
    return sha256(row_hashes.values.tobytes()).hexdigest()


def _executor(jobname, cache_arguments, pre_execute, commands, post_execute,
              post_cache=None, post_cache_arguments=dict(),
              dry=True, use_grid=True, ppn=10, nocache=False,
              pmem='20GB', environment=settings.QIIME_ENV, walltime='4:00:00',
              wait=True, timing=True, verbose=stderr, array=1,
              dirty=False, load_cachefile=None):
    """

    Parameters
    ----------
    jobname : str
    cache_arguments : []
    pre_execute : function
    commands : [] or dict:{'pre': [], 'main': [], 'post': []}
    post_execute : function
    post_cache : function
        A function that is called, after results have been loaded from cache /
        were generated. E.g. drawing rarefaction curves.
    environment : str

    ==template arguments that should be copied to calling analysis function==
    dry : bool
        Default: True.
        If True: only prepare working directory and create necessary input
        files and print the command that would be executed in a non dry run.
        For debugging. Workdir is not deleted.
        "pre_execute" is called, but not "post_execute".
    use_grid : bool
        Default: True.
        If True, use qsub to schedule as a grid job, otherwise run locally.
    nocache : bool
        Default: False.
        Normally, successful results are cached in .anacache directory to be
        retrieved when called a second time. You can deactivate this feature
        (useful for testing) by setting "nocache" to True.
    wait : bool
        Default: True.
        Wait for results.
    walltime : str
        Default: "12:00:00".
        hh:mm:ss formated wall runtime on cluster.
    ppn : int
        Default: 10.
        Number of CPU cores to be used.
    pmem : str
        Default: '8GB'.
        Resource request for cluster jobs. Multiply by ppn!
    timing : bool
        Default: True
        Use '/usr/bin/time' to log run time of commands.
    verbose : stream
        Default: sys.stderr
        To silence this function, set verbose=None.
    array : int
        Default: 1 = deactivated.
        Only for Torque submits: make the job an array job.
        You need to take care of correct use of ${PBS_JOBID} !
    dirty : bool
        Defaul: False.
        If True, temporary working directory will not be removed.
    load_cachefile : str
        Default: None
        Provide a filepath to an existing cache file and the function will
        load data from there - even if cache file is for a different type of
        analysis!

    Returns
    -------
    """
    DIR_CACHE = '.anacache'
    FILE_STATUS = 'finished.info'
    results = {'results': None,
               'workdir': None,
               'qid': None,
               'file_cache': None,
               'cached_inputs': dict(),
               'timing': None,
               'cache_version': 20260622,
               'created_on': None,
               'conda_env': 'unknown',
               'jobname': jobname}

    # create an ID function if no post_cache function is supplied
    def _id(x):
        return x
    if post_cache is None:
        post_cache = _id

    if load_cachefile is not None:
        # a short cut to load a specific cache file, ignoring all the remaining
        # mechanisms!!
        if verbose:
            verbose.write("Loading existing results from '%s'. \n" %
                          load_cachefile)
        f = open(load_cachefile, 'rb')
        results = load(f)
        f.close()
        if results['jobname'] != jobname:
            verbose.write("!!Warning: loaded cache is for analysis '%s', but you are using it as '%s'!!" % (results['jobname'], jobname))
        return post_cache(results, **post_cache_arguments)


    # phase 1: compute signature for cache file
    # convert skbio.DistanceMatrix object to a sorted version of its data for
    # hashing
    cache_args_original = dict()
    for arg in cache_arguments.keys():
        if type(cache_arguments[arg]) == DistanceMatrix:
            cache_args_original[arg] = cache_arguments[arg]
            dm = cache_arguments[arg]
            cache_arguments[arg] = dm.filter(sorted(dm.ids)).data
        if (type(cache_arguments[arg]) == dict):
            if (len({type(v) for v in cache_arguments[arg].values()} ^
                    set([DistanceMatrix])) == 0):
                cache_args_original[arg] = cache_arguments[arg]
                cache_arguments[arg] = OrderedDict(
                    {k: dm.filter(sorted(dm.ids)).data
                     for k, dm
                     in cache_arguments[arg].items()})
        if (type(cache_arguments[arg]) == pd.Series) or (type(cache_arguments[arg]) == pd.DataFrame):
            cache_args_original[arg] = cache_arguments[arg]
            cache_arguments[arg] = pandas_hash(cache_arguments[arg])
        if isinstance(cache_arguments[arg], Table):
            cache_args_original[arg] = cache_arguments[arg]
            cache_arguments[arg] = sorted(list(cache_arguments[arg].ids('sample'))) + \
                                   sorted(list(cache_arguments[arg].ids('observation'))) + \
                                   [cache_arguments[arg].get_table_density()]
        if (type(cache_arguments[arg]) == str) and os.path.exists(cache_arguments[arg]):
            cache_args_original[arg] = cache_arguments[arg]
            if os.path.isfile(cache_arguments[arg]):
                # if argument can be used as a file path and the file actually exists...
                # ... than use the md5sum of the file instead of the filepath for cache fingerprint
                # Thus, moving the file will not affect the cache fingerprint
                cache_arguments[arg] = _md5(cache_arguments[arg])
            else:
                # assume path is directory
                cache_arguments[arg] = os.path.abspath(cache_arguments[arg])

        # for better debugging, write hash sum for each input argument in result object
        if cache_arguments[arg] is None:
            results['cached_inputs'][arg] = None
        else:
            results['cached_inputs'][arg] = md5(str(cache_arguments[arg]).encode()).hexdigest()

    _input = OrderedDict(sorted(cache_arguments.items()))
    results['file_cache'] = "%s/%s.%s" % (DIR_CACHE, md5(
        str(_input).encode()).hexdigest(), jobname)

    # convert back cache arguments if necessary
    for arg in cache_args_original.keys():
        cache_arguments[arg] = cache_args_original[arg]

    # phase 2: if cache contains matching file, load from cache and return
    if os.path.exists(results['file_cache']) and (nocache is not True):
        if verbose:
            verbose.write("Using existing results from '%s'. \n" %
                          results['file_cache'])
        f = open(results['file_cache'], 'rb')
        results = load(f)
        f.close()
        return post_cache(results, **post_cache_arguments)

    # phase 3: search in TMP dir if non-collected results are
    # ready or are waited for
    dir_tmp = gettempdir()
    if use_grid:
        dir_tmp = os.environ['HOME'] + '/TMP/'
        if not os.path.exists(dir_tmp):
            raise ValueError('Temporary directory "%s" does not exist. '
                             'Please create it and restart.' % dir_tmp)

    # collect all tmp workdirs that contain the right cache signature
    pot_workdirs = []
    for _dir in next(os.walk(dir_tmp))[1]:
        # a potential working directory needs to have the matching job name
        if _dir.startswith('ana_%s_' % results['jobname']):
            # for shared computers, make sure you have permission to read dir contents
            if not os.access(os.path.join(dir_tmp, _dir), os.R_OK):
                continue
            potwd = os.path.join(dir_tmp, _dir)
            # and a matching cache file signature
            if results['file_cache'].split('/')[-1] in next(os.walk(potwd))[2]:
                pot_workdirs.append(potwd)
    finished_workdirs = []
    for wd in pot_workdirs:
        all_finished = os.path.exists('%s/finished.info' % wd)
        # for i in range(array):
        #     exp_finish_suffix = ""
        #     if array > 1:
        #         exp_finish_suffix = str(int(i+1))
        #     if (array == 1):
        #         if (settings.GRIDNAME == 'JLU'):
        #             if use_grid:
        #                 exp_finish_suffix = 'undefined'
        #             else:
        #                 exp_finish_suffix = '1'
        #         else:
        #             exp_finish_suffix = '1'
        #     if not os.path.exists('%s/finished.info%s' % (wd, exp_finish_suffix)):
        #         all_finished = False
        #         break
        if all_finished:
            finished_workdirs.append(wd)
    if len(pot_workdirs) > 0 and len(finished_workdirs) <= 0:
        if verbose:
            verbose.write(
                ('Found %i temporary working directories, but non of '
                 'them have finished (missing "finished.info" file). If no job is currently running,'
                 ' you might want to delete these directories and res'
                 'tart:\n  %s\n') % (len(pot_workdirs),
                                     "\n  ".join(pot_workdirs)))
        return results

    if len(finished_workdirs) > 0:
        # arbitrarily pick first found workdir
        results['workdir'] = finished_workdirs[0]
        if verbose:
            verbose.write('found matching working dir "%s"\n' %
                          results['workdir'])
    else:
        # create a temporary working directory
        prefix = 'ana_%s_' % jobname
        results['workdir'] = mkdtemp(prefix=prefix, dir=dir_tmp)
        if verbose:
            verbose.write("Working directory is '%s', cachefile is '%s'. " %
                          (results['workdir'], results['file_cache']))
        # leave an empty file in workdir with cache file name to later
        # parse results from tmp dir
        f = open("%s/%s" % (results['workdir'],
                            results['file_cache'].split('/')[-1]), 'w')
        f.close()

        pre_execute(results['workdir'], cache_arguments)

        lst_commands = commands(results['workdir'], ppn, cache_arguments)
        # convert to new dict structure instead of flat list
        if isinstance(lst_commands, list):
            lst_commands = {'main': lst_commands, 'pre': [], 'post': []}
        # device creation of a file _after_ execution of the job in workdir
        final_cmd = 'touch %s/%s' % (results['workdir'], FILE_STATUS)
        lst_commands['post'].append(final_cmd)

        results['qid'] = cluster_run(
            lst_commands, 'ana_%s' % jobname, results['workdir']+'mock',
            environment, ppn=ppn, wait=wait, dry=dry,
            pmem=pmem, walltime=walltime,
            file_qid=results['workdir']+'/cluster_job_id.txt',
            file_condaenvinfo=results['workdir']+'/conda_info.txt',
            timing=timing,
            file_timing=results['workdir']+('/timing${%s}.txt' % settings.VARNAME_PBSARRAY),
            array=array, use_grid=use_grid)
        if dry:
            return results
        if wait is False:
            return results

    results['results'] = post_execute(results['workdir'],
                                      cache_arguments)
    results['created_on'] = datetime.fromtimestamp(
        time()).strftime('%Y-%m-%d %H:%M:%S')

    results['conda_env'] = environment
    if environment is not None:
        with open(results['workdir']+'/conda_info.txt', 'r') as f:
            results['conda_list'] = f.readlines()

    results['timing'] = []
    for timingfile in next(os.walk(results['workdir']))[2]:
        if timingfile.startswith('timing'):
            with open(results['workdir']+'/'+timingfile, 'r') as content_file:
                results['timing'] += content_file.readlines()

    if results['results'] is not None:
        if not dirty:
            shutil.rmtree(results['workdir'])
            if verbose:
                verbose.write(" Was removed.\n")

    os.makedirs(os.path.dirname(results['file_cache']), exist_ok=True)
    f = open(results['file_cache'], 'wb')
    dump(results, f)
    f.close()

    return post_cache(results, **post_cache_arguments)

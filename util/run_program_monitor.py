import asyncio
import logging
import shlex
from argparse import ArgumentParser
from asyncio import StreamWriter, StreamReader
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Iterable

from util import run, setup_logging

log = logging


class MonitoredProgram:
    run_command: Iterable[str]
    input: str
    cwd: str
    time_out: int
    allow_non_zero_exit_status: bool
    start_time: datetime = None
    stop_time: datetime = None
    program_output: str = None

    def __init__(self, run_command: str, input='', cwd: str = None, time_out: int = None,
                 allow_non_zero_exit_status=False) -> None:
        self.run_command = shlex.split(run_command)
        self.input = input
        self.cwd = cwd
        self.time_out = time_out
        self.allow_non_zero_exit_status = allow_non_zero_exit_status

    async def monitor_running_process(self):
        process = await asyncio.create_subprocess_exec(
            *self.run_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd
        )
        stdin = process.stdin
        stdout = process.stdout
        stderr = process.stderr

        # Instead, show just the launch command.
        log.info("$ " + " ".join(self.run_command))

        loop = asyncio.get_event_loop()
        logStdout = loop.create_task(
            self.update_from_container_logs(log, stdin, stdout, True, 'stdout', record_to_program_output=True,
                                            level=logging.INFO)
        )
        logStderr = loop.create_task(
            self.update_from_container_logs(log, stdin, stderr, False, 'stderr', level=logging.INFO)
        )
        waitContainerCompletion = loop.create_task(
            self.wait_for_docker_run_completion(process, log)
        )
        done, pending = await asyncio.wait(
            {logStdout,
             logStderr,
             waitContainerCompletion},
            return_when=asyncio.ALL_COMPLETED
        )

        if len(pending) != 0:
            raise RuntimeError(
                'Not all run tasks completed. There were {} unexpected pending tasks.'.format(len(pending)))
        # This will raise exceptions if any of the tasks raised an exception. Actual result is not interesting
        map(lambda completed: completed.result(), done)

        log.info(f'[Monitor]: Program runtime: {self.stop_time - self.start_time}')

        if not self.allow_non_zero_exit_status and self.exitCode != 0:
            raise RuntimeError('Process exited with exit code: {}'.format(self.exitCode))

    async def wait_for_docker_run_completion(self, process: asyncio.subprocess.Process, log):
        try:
            log.info("[Monitor]: Setting task timeout to {}".format(self.time_out))
            await asyncio.wait_for(process.wait(), self.time_out)
            self.exitCode = process.returncode
        except asyncio.TimeoutError:
            log.info("[Monitor]: Process timeout exceeded. Killing process.")
            process.kill()
            log.info("[Monitor]: Process killed.")
            self.exitCode = -1

        log.info("[Monitor]: Exit code: " + str(self.exitCode))

    async def wait_for_container_completion(self, container, log):
        waitResponse = container.wait(timeout=self.time_out)
        # If we get here the container is either not running anymore or the timeout expired.
        if container.status != 'exited':
            log.error('[Monitor]: Timeout for docker run, stopping.')
            container.stop()
        if waitResponse['Error'] is not None:
            log.error("[Monitor]: Wait response error: " + waitResponse['Error'])
        self.exitCode = waitResponse['StatusCode']
        log.info("[Monitor]: Exit code: " + str(self.exitCode))

    async def update_from_container_logs(self, log, stdin: StreamWriter, stdout: StreamReader, provide_input: bool,
                                         target: str, record_to_program_output=False, level=logging.DEBUG):
        submitted_input = False
        record_output = True
        self.program_output = ''
        self.start_time = datetime.now()
        # log.info(f'[Monitor]: Starting program run at {self.start_time}')
        while True:
            log_line = await stdout.readline()
            log_line = log_line.decode().strip("\n")
            if not log_line:
                break

            if record_output:
                log.log(level, f"[{target.rjust(7)}]: {log_line}")
                if record_to_program_output:
                    self.program_output += log_line + '\n'
            else:
                log.error(
                    '[Monitor]: Process provided additional output after submitting solution, will ignore: ' + log_line)

            if 'START' == log_line:
                if not provide_input:
                    log.error("[Monitor]: Tried to get input from wrong stream " + target)
                elif submitted_input:
                    log.error(
                        '[Monitor]: Tried to get input more than once. Probably an attempt to manipulate timing.')
                else:
                    submitted_input = True
                    self.start_time = datetime.now()
                    # Submit problem to process
                    stdin.write(self.input.encode())
                    stdin.close()
                    log.info(f"[{'stdin'.rjust(7)}]: {self.input}")
                    log.info('[Monitor]: Problem was submitted to process at ' + str(self.start_time))
            if 'STOP' == log_line:
                if not provide_input:
                    log.error("[Monitor]: Tried to finish output from wrong stream " + target)
                if self.stop_time is not None:
                    log.error(
                        '[Monitor]: Tried to stop the timer more than once. Probably an attempt to manipulate timing.')
                    # Conservatively, use the last DONE as reference (resulting in the longest time)
                    self.stop_time = datetime.now()
                else:
                    self.stop_time = datetime.now()
                    record_output = False
                    log.info('[Monitor]: Process finished the problem at ' + str(self.stop_time))

        exit_time = datetime.now()
        # log.info(f'[Monitor]: Process exited at {exit_time}')
        if self.stop_time is None:
            self.stop_time = exit_time

    def run(self):
        asyncio.get_event_loop().run_until_complete(self.monitor_running_process())


if __name__ == '__main__':
    setup_logging()
    parser = ArgumentParser(description='Tool to test BEAST practical course student solutions')

    parser.add_argument('--compile-command', type=str, default='make', help='The command to run to build the solution.')
    parser.add_argument('--clean-command', type=str, default='make clean', help='Command to run after running program.')
    parser.add_argument('--timeout', type=int,
                        help='Set a timeout for compilation and program run separately after which execution is aborted.')
    parser.add_argument('--reference-output', type=str, help='Reference output file to compare solution to.')
    parser.add_argument('--use-numdiff', type=bool, default=False,
                        help='Use numdiff program to compare output rather than diff.')
    parser.add_argument('--numdiff-relative-error', type=float, help='Maximum relative error to accept for numdiff')
    parser.add_argument('--numdiff-absolute-error', type=float, help='Maximum absolute error to accept for numdiff')
    parser.add_argument('--numdiff-additional-arguments', type=str, default='', help='Additional arguments to pass to numdiff')
    parser.add_argument('solution_directory', type=str,
                        help="The directory containing the student solution and the necessary build files.")
    parser.add_argument('run_command', type=str, help='Command to execute.')

    result = parser.parse_args()

    try:
        logging.info('Now compiling your program')
        run(result.compile_command, cwd=result.solution_directory, timeout=result.timeout)

        monitor = MonitoredProgram(result.run_command, cwd=result.solution_directory, input='',
                                   time_out=result.timeout)
        monitor.run()

        run(result.clean_command, cwd=result.solution_directory, timeout=result.timeout)

        if result.reference_output:
            program_output_file = NamedTemporaryFile(delete=False)
            try:
                program_output_file.write(monitor.program_output.encode())
                program_output_file.close()

                if not result.use_numdiff:
                    program_output = run(
                        f'diff -y {result.reference_output} {program_output_file.name}',
                        cwd=result.solution_directory
                    )
                else:
                    log.info('Reference Output'.ljust(49, ' ') + '| Your Output')
                    program_output = run(
                        f'numdiff --overview=130 --minimal --statistics '
                        f'{("-r " + str(result.numdiff_relative_error) + " ") if result.numdiff_relative_error is not None else ""}'
                        f'{("-a " + str(result.numdiff_absolute_error) + " ") if result.numdiff_absolute_error is not None else ""}'
                        f' {result.numdiff_additional_arguments} '
                        f'"{result.reference_output}" "{program_output_file.name}"',
                        cwd=result.solution_directory
                    )
                log.info('Your submission was successfully verified against the reference implementation.')

            finally:
                program_output_file.close()
    except RuntimeError as e:
        logging.error(e)
        exit(-1)

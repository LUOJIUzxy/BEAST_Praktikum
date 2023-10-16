import logging
import subprocess
import sys
from pathlib import Path

import os


def run(program, input='', cwd=None, shell=True, exit_code=0, print_output=True, timeout=None):
    if timeout is not None:
        program = f"timeout {timeout} {program}"

    if cwd is None:
        cwd = os.getcwd()
    logging.info(f'{Path(cwd).name} > {program}')

    output = ''
    process = subprocess.Popen(
        program,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        cwd=cwd,
        shell=shell
    )

    if input:
        process.stdin.write(input.encode('utf-8'))

    while True:
        line = process.stdout.readline().decode('utf-8')
        if line == '' and process.poll() is not None:
            break
        if line:
            if print_output:
                logging.info(f'| {line.strip()}')
            output += line
    actual_exit_code = process.poll()

    if exit_code != actual_exit_code:
        raise RuntimeError(f"Program {program} exited with unexpected output code {actual_exit_code}.")

    return output


def setup_logging(log_file=None, log_invocation_command=True):
    handlers = [
        logging.StreamHandler()
    ]
    if log_file:
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),

    # https://youtrack.jetbrains.com/issue/PY-39762
    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=handlers
    )
    if log_invocation_command:
        logging.info(f'> {" ".join(sys.argv)}')

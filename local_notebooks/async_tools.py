import sys
import asyncio

async def stream_reader(stream, id, to):
    id = '' if id is None else f'{id}. '
    data = b''
    while True:
        read = await stream.read(n=4096)
        if not read: break
        if to is not None:
            *complete_lines, data = (data + read + b'X').splitlines()
            data = data[:-1]
            for line in complete_lines:
                line = line.rstrip()
                if line: print(f"{id}{line.decode('utf-8')}", file=to, end='\n', flush=True)

async def wait_for_subprocess(subprocess, print_output=False, id=None):
    await asyncio.gather(
            stream_reader(subprocess.stdout, id, (sys.stdout if print_output else None)),
            stream_reader(subprocess.stderr, id, (sys.stderr if print_output else None)),
        )
    return await subprocess.wait()

async def wait_for_subprocesses(*processes, print_output=False):
    return await asyncio.gather(*[wait_for_subprocess(p, print_output=print_output, id=i if len(processes)>1 else None) for i, p in enumerate(processes)])
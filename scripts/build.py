#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def run(argv, env):
    command = [str(value) for value in argv]
    print('+ ' + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def record_build(kind, construction):
    directory = ROOT / 'dist' / kind
    files = ('Rover', 'libcalculator.so') if kind == 'android' else ('libcalculator_skel.so',)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    record = {
        'abi': 2 if construction else 1,
        'files': {name: digest(directory / name) for name in files},
        'limits_sha256': digest(ROOT / 'config/rover_limits.h'),
        'idl_sha256': digest(ROOT / 'inc/calculator.idl'),
    }
    temporary = directory / 'build.json.tmp'
    temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    temporary.replace(directory / 'build.json')


def build_host(cmake, jobs, env):
    output = ROOT / 'build/host'
    run([
        cmake, '-S', ROOT, '-B', output,
        '-DROVER_HOST_TOOLS=ON', '-DCMAKE_BUILD_TYPE=Release',
        f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={ROOT / "dist/host"}',
    ], env)
    run([cmake, '--build', output, '--parallel', jobs], env)


def build_device(kind, cmake, sdk, jobs, construction, env):
    output = ROOT / 'build' / kind
    artifacts = ROOT / 'dist' / kind
    (artifacts / 'build.json').unlink(missing_ok=True)
    options = [
        '-S', ROOT, '-B', output, '-G', 'Ninja',
        f'-DHEXAGON_SDK_ROOT={sdk}',
        f'-DHEXAGON_CMAKE_ROOT={sdk / "build/cmake"}',
        f'-DCMAKE_MAKE_PROGRAM={sdk / "tools/cmake-3.28.3-linux-x86_64/bin/ninja"}',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DROVER_ENABLE_BUILD=' + ('ON' if construction else 'OFF'),
        f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={artifacts}',
        f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={artifacts}',
        f'-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={artifacts}',
    ]
    env = env.copy()
    if kind == 'android':
        ndk = sdk / 'tools/android-ndk-r25c'
        env['ANDROID_ROOT_DIR'] = str(ndk)
        options += [
            f'-DCMAKE_TOOLCHAIN_FILE={sdk / "build/cmake/android_toolchain.cmake"}',
            '-DV=android_ReleaseG_aarch64', '-DANDROID_ABI=arm64-v8a',
            '-DANDROID_NATIVE_API_LEVEL=26', f'-DANDROID_NDK={ndk}',
            '-DANDROID_STL=none', '-DCMAKE_SYSTEM_NAME=Android',
            '-DDSP_TYPE=3', '-DOS_TYPE=HLOS', '-DPREBUILT_LIB_DIR=android_aarch64',
        ]
    else:
        toolchain = sdk / 'tools/HEXAGON_Tools/19.0.04'
        env['HEXAGON_TOOLS_ROOT'] = str(toolchain)
        env['DEFAULT_HEXAGON_TOOLS_ROOT'] = str(toolchain)
        options += [
            f'-DCMAKE_TOOLCHAIN_FILE={sdk / "build/cmake/hexagon_toolchain.cmake"}',
            '-DV=hexagon_ReleaseG_toolv19_v75', '-DADD_SYMBOLS=1',
            '-DDO_SYSTEM_INCLUDE=0', '-DDSP_VERSION=v75',
            f'-DHEXAGON_TOOLS_ROOT={toolchain}',
            '-DPREBUILT_LIB_DIR=hexagon_toolv19_v75', '-DQURT_OS=0', '-DSIM_TYPE=',
        ]
    run([cmake, *options], env)
    run([cmake, '--build', output, '--parallel', jobs], env)
    record_build(kind, construction)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compile Rover and host tools without deploying or running them.')
    parser.add_argument('--target', choices=('all', 'android', 'hexagon', 'host'), default='all')
    parser.add_argument('--jobs', type=positive_integer, default=4)
    parser.add_argument('--sdk', type=Path, default=os.environ.get('HEXAGON_SDK_ROOT') or None,
                        help='SDK directory; defaults to HEXAGON_SDK_ROOT')
    parser.add_argument('--with-build', action='store_true',
                        help='Include graph construction (automatic on graph-build-insert)')
    args = parser.parse_args(argv)
    sdk = Path(args.sdk).expanduser().resolve() if args.sdk else None
    if args.target != 'host' and sdk is None:
        parser.error('Set HEXAGON_SDK_ROOT or --sdk')
    if sdk is not None and not sdk.is_dir():
        parser.error(f'SDK directory does not exist: {sdk}')
    capabilities = json.loads((ROOT / 'config/capabilities.json').read_text())
    construction = args.with_build or capabilities['construction']
    if construction and args.target != 'host' and not (ROOT / 'src/build/rover_insert_build_core.c').is_file():
        parser.error('Checkout graph-build-insert to compile construction')

    env = os.environ.copy()
    cmake = env.get('CMAKE') or 'cmake'
    if sdk is not None:
        cmake = sdk / 'tools/cmake-3.28.3-linux-x86_64/bin/cmake'
        env['HEXAGON_SDK_ROOT'] = str(sdk)
    if args.target in ('all', 'host'):
        build_host(cmake, args.jobs, env)
    for kind in ('android', 'hexagon'):
        if args.target in ('all', kind):
            build_device(kind, cmake, sdk, args.jobs, construction, env)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print('Rover build: ' + str(error), file=sys.stderr)
        raise SystemExit(error.returncode if error.returncode > 0 else 1)
    except OSError as error:
        print('Rover build: ' + str(error), file=sys.stderr)
        raise SystemExit(1)

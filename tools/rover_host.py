#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import struct
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=(
        'Host command entry point; paths are local unless explicitly called device-*.\n'
        '\n'
        'Importing this module never builds, deploys, starts a device task or loads data.\n'
    ))
    p.add_argument('command', choices=('search', 'build-index', 'insert', 'resume', 'prepare', 'push'))
    p.add_argument('--serial')
    p.add_argument('--profile')
    p.add_argument('--graph-entry',type=int)
    p.add_argument('--build-distance',choices=('cpu','dsp'),default='cpu')
    p.add_argument('--device-dir', default='/data/local/tmp/Rover')
    for name in ('index-dir', 'output-index-dir', 'base', 'query', 'gt', 'graph', 'parent-index', 'checkpoint', 'results-dir'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--method', choices=('sq', 'pq'))
    p.add_argument('--sq-bits', type=int, choices=(4, 8))
    p.add_argument('--pq-bits', type=int, choices=(4, 8))
    grouping = p.add_mutually_exclusive_group()
    for name in ('pq-m', 'pq-subdim', 'pq-compression'):
        grouping.add_argument('--'+name, type=int)
    p.add_argument('--residual-bits', type=int, choices=(4, 8))
    p.add_argument('--third-residual', choices=('off', 'on'), default=None)
    p.add_argument('--correction', choices=('off', 'online'), default='online')
    p.add_argument('--refine', choices=('off', 'on'), default='off')
    refine = p.add_mutually_exclusive_group()
    refine.add_argument('--refine-factor', type=int, default=2)
    refine.add_argument('--refine-all', action='store_true')
    p.add_argument('--storage', choices=('memory', 'disk'), default='disk')
    p.add_argument('--search-mode', choices=('sqmd', 'db-pair'), default='sqmd')
    p.add_argument('--build-method', choices=('stream-insert', 'search-build', 'rnn-descent'), default='stream-insert')
    p.add_argument('--graph-format', choices=('rover', 'dense', 'nsg', 'nndescent'), default='nndescent')
    for name, default in dict(top_k=10, pool=128, entry_points=32, lanes=4, source_window=1,
                              first_stage_factor=3, cache_pages=2048, refine_cache_pages=2048,
                              pq_train_samples=100000, pq_train_iters=20, seed=42,
                              build_pool=128, build_candidates=16, build_iterations=1,
                              build_threads=1, build_refine_candidates=0, backlink_threads=1,
                              reverse_candidates=0, checkpoint_every=1000000, cooldown_every=0, cooldown_ms=10000, domain=3,
                              unsigned_pd=1, chunk_vectors=8192, host_threads=8,
                              memory_budget_mb=512, cluster_iters=20, expand_count=32, rnn_s=16, rnn_t1=4).items():
        p.add_argument('--'+name.replace('_', '-'), type=int, default=default)
    p.add_argument('--graph-degree', type=int)
    p.add_argument('--num-clusters', type=int)
    p.add_argument('--residual-layout', choices=('a','b','c','d'), default='a')
    p.add_argument('--reorder', choices=('off', 'on'), default='off')
    p.add_argument('--cn', choices=('off', 'on'), default='on')
    p.add_argument('--mnn', choices=('off', 'on'), default='on')
    p.add_argument('--split-distance', choices=('off', 'on'), default='off')
    p.add_argument('--reencode', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Print the requested task; perform no preparation or device operations')
    return p


class Device:
    def __init__(self, serial, directory):
        path = PurePosixPath(directory)
        if not path.is_absolute() or '..' in path.parts or str(path) in ('/', '/data', '/data/local', '/data/local/tmp'):
            raise ValueError('device-dir must be a dedicated absolute application directory')
        self.root = str(path)
        self.adb = ['adb'] + (['-s', serial] if serial else [])

    def call(self, *args, **kwargs):
        return subprocess.run(self.adb+list(map(str, args)), check=True, **kwargs)

    def shell(self, argv, **kwargs):
        return self.call('shell', shlex.join(list(map(str, argv))), **kwargs)

    def mkdir(self, path):
        self.shell(['mkdir', '-p', path])

    def push(self, local, remote):
        self.call('push', Path(local).resolve(), remote)

    def binaries(self):
        artifacts = (ROOT/'dist/android/Rover', ROOT/'dist/android/libcalculator.so', ROOT/'dist/hexagon/libcalculator_skel.so')
        if not all(p.is_file() for p in artifacts):
            raise ValueError('Compile Rover and its DSP library before deployment')
        records=[json.loads((ROOT/'dist'/kind/'build.json').read_text()) for kind in ('android','hexagon')]
        expected_abi=2 if json.loads((ROOT/'config/capabilities.json').read_text())['construction'] else 1
        if any(item['abi']!=expected_abi for item in records) or any(records[0][k]!=records[1][k] for k in ('limits_sha256','idl_sha256')):
            raise ValueError('CPU/DSP build configuration mismatch; rebuild both for this branch')
        for key, source in (('limits_sha256', 'config/rover_limits.h'), ('idl_sha256', 'inc/calculator.idl')):
            if records[0][key] != hashlib.sha256((ROOT/source).read_bytes()).hexdigest():
                raise ValueError('Shared configuration changed; rebuild CPU and DSP')
        for kind,item in zip(('android','hexagon'),records):
            for name,digest in item['files'].items():
                if hashlib.sha256((ROOT/'dist'/kind/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Build artifact changed: '+name)
        self.mkdir(self.root+'/bin')
        for artifact in artifacts:
            self.push(artifact, self.root+'/bin/'+artifact.name)
        self.shell(['chmod', '755', self.root+'/bin/Rover'])


def validate_options(args):
    if args.reorder == 'on' and args.cn == 'off' and args.mnn == 'off':
        raise ValueError('Graph reordering requires at least one of --cn or --mnn on')
    positive = ('top_k', 'pool', 'entry_points', 'first_stage_factor', 'refine_factor',
                'pq_train_samples', 'pq_train_iters', 'build_pool', 'build_candidates',
                'build_iterations', 'build_threads', 'backlink_threads', 'chunk_vectors',
                'host_threads', 'memory_budget_mb', 'cluster_iters', 'expand_count', 'rnn_s', 'rnn_t1')
    for name in positive:
        if not 1 <= getattr(args, name) <= 2147483647:
            raise ValueError(name+' must be a positive 32-bit integer')
    for name in ('seed', 'build_refine_candidates', 'reverse_candidates', 'checkpoint_every', 'cooldown_every', 'cooldown_ms'):
        if not 0 <= getattr(args, name) <= 2147483647:
            raise ValueError(name+' must be a nonnegative 32-bit integer')
    if not 1 <= args.top_k <= 100 or not args.top_k <= args.pool <= 2000:
        raise ValueError('top-k must be 1..100; pool must be top-k..2000')
    if args.lanes not in range(1, 9) or args.source_window not in (1, 2) or not 1 <= args.expand_count <= 32:
        raise ValueError('lanes must be 1..8; source-window 1/2; expand-count 1..32')
    if args.first_stage_factor*args.top_k > 2000 or args.refine_factor*args.top_k > 2000:
        raise ValueError('Candidate factors exceed the compiled capacity of 2000')
    for name in ('cache_pages', 'refine_cache_pages'):
        value = getattr(args, name)
        if value < 1 or value > 65536 or value & (value-1):
            raise ValueError(name+' must be a power of two in 1..65536')
    if args.domain not in range(4) or args.unsigned_pd not in (0, 1):
        raise ValueError('domain must be 0..3 and unsigned-pd must be 0/1')
    construction = args.command in ('build-index', 'insert', 'resume')
    if args.cooldown_every and (not construction or args.build_method != 'search-build' or not args.cooldown_ms):
        raise ValueError('Cooldown requires search-build and positive cooldown-ms')
    if args.command == 'search' and not args.query:
        raise ValueError('search requires --query')
    if construction:
        if not args.base:
            raise ValueError('Construction requires the complete --base dataset')
        if args.graph:
            raise ValueError('Use --parent-index with insert; construction creates its output graph')
        if args.reorder == 'on' or args.residual_layout != 'a':
            raise ValueError('Apply graph/residual reordering after construction')
        max_threads=64 if args.build_method == 'rnn-descent' else 4
        if args.build_threads > max_threads or args.backlink_threads > 4 or args.build_pool > 2000:
            raise ValueError(f'Build threads must be 1..{max_threads}, backlink threads 1..4, build-pool at most 2000')
        if args.build_candidates > args.build_pool:
            raise ValueError('build-candidates must not exceed build-pool')
        if args.reverse_candidates > args.build_candidates:
            raise ValueError('reverse-candidates must not exceed build-candidates')
        if args.build_refine_candidates and not args.build_candidates <= args.build_refine_candidates <= args.build_pool:
            raise ValueError('build-refine-candidates must lie between candidates and pool')
        if args.build_method == 'stream-insert' and args.build_threads != 1:
            raise ValueError('stream-insert requires --build-threads 1')
        if args.command in ('insert', 'resume') and args.build_method != 'stream-insert':
            raise ValueError('insert/resume require stream-insert')
    if args.parent_index and args.command != 'insert':
        raise ValueError('--parent-index is only accepted by insert')
    if args.checkpoint and args.command != 'resume':
        raise ValueError('Use resume to load an existing checkpoint')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p=parser()
    initial=p.parse_args(argv)
    if initial.profile:
        profile=Path(initial.profile)
        if not profile.is_file():profile=ROOT/'config/profiles'/(initial.profile+'.json')
        defaults=json.loads(profile.read_text())
        actions={a.dest:a for a in p._actions if a.dest not in ('help', 'command', 'profile')}
        if not isinstance(defaults, dict) or set(defaults)-set(actions):
            raise ValueError('Unknown profile fields')
        for group in (('pq_m', 'pq_subdim', 'pq_compression'), ('refine_factor', 'refine_all')):
            if any(token.split('=', 1)[0] == '--'+name.replace('_', '-') for token in argv for name in group):
                for name in group:
                    defaults.pop(name, None)
        for name, value in defaults.items():
            action=actions[name]
            if action.type is int and type(value) is not int:
                raise ValueError('Profile field must be an integer: '+name)
            if isinstance(action, argparse._StoreTrueAction) and type(value) is not bool:
                raise ValueError('Profile field must be boolean: '+name)
            if action.choices and value not in action.choices:
                raise ValueError('Invalid profile choice: '+name)
            if action.type is Path:
                defaults[name]=Path(value)
        p.set_defaults(**defaults)
    args=p.parse_args(argv)
    if args.dry_run:
        print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
        return 0
    if args.command == 'push':
        Device(args.serial, args.device_dir).binaries()
        return 0
    validate_options(args)
    construction = args.command in ('build-index', 'insert', 'resume')
    capabilities = json.loads((ROOT/'config/capabilities.json').read_text())
    if construction and not capabilities['construction']:
        raise ValueError('Checkout graph-build-insert for construction')
    if not (args.index_dir or args.output_index_dir):
        raise ValueError('Specify --index-dir or --output-index-dir')
    if args.refine == 'on' and args.correction == 'off':
        raise ValueError('Coarse-only search with final residual refinement is excluded from this release')
    if construction and args.reorder == 'on':
        raise ValueError('Reorder a completed index through the search/import workflow')
    if args.command == 'resume' and (not args.checkpoint or not args.checkpoint.is_file()):
        raise ValueError('resume requires a local --checkpoint file')
    if args.command == 'insert' and not args.parent_index:
        raise ValueError('insert requires --parent-index and the complete prefix+new-row --base dataset')
    if args.third_residual is None:
        manifest = (args.output_index_dir or args.index_dir)/'index.json'
        args.third_residual = 'on' if manifest.is_file() and json.loads(manifest.read_text()).get('third_residual') else 'off'
    from index_bundle import prepare, validate, record, atomic_json, vector_shape, read_graph
    directory, data, base = prepare(args, construction=construction or args.command=='prepare')
    if args.command == 'prepare':
        print(directory/'index.json')
        return 0
    n, dim = data['n'], data['dim']
    if (not construction and args.top_k > n) or args.entry_points > n:
        raise ValueError('top-k and entry-points must not exceed N')
    if args.method == 'pq' and args.search_mode == 'db-pair':
        raise ValueError('DB Pair supports SQ; select sqmd for PQ')
    if args.split_distance == 'on' and (args.method != 'sq' or args.sq_bits != 4 or args.correction != 'online' or not data['files'].get('bias')):
        raise ValueError('Split distance requires SQ4 online correction and a prepared bias file')
    if construction and args.build_method != 'rnn-descent' and args.build_candidates < data['graph_degree']:
        raise ValueError('build-candidates must cover graph-degree')
    if construction and args.build_method == 'rnn-descent' and args.rnn_s > data['graph_degree']:
        raise ValueError('rnn-s must not exceed graph-degree')
    if construction and data['files'].get('graph'):
        raise ValueError('The output bundle already contains a graph; use a fresh output directory')
    if construction and (base is None or __import__('generate_final_refine').sha(base) != data['raw_sha256']):
        raise ValueError('Construction base must match the prepared representation')
    gt_k = None
    if not construction:
        if args.query is None:
            raise ValueError('search requires --query')
        nq, qdim = vector_shape(args.query)
        if qdim != dim:
            raise ValueError('Query dimension differs from index')
        if args.gt:
            with args.gt.open('rb') as f:
                header=f.read(4)
            if len(header) != 4:
                raise ValueError('Truncated ground truth header')
            gt_k=struct.unpack('<i', header)[0]
            if not args.top_k <= gt_k <= 1000000 or args.gt.stat().st_size < nq*(gt_k+1)*4:
                raise ValueError('Ground truth does not cover query count/top-k')
    parent = None
    if args.parent_index:
        parent = validate(args.parent_index, third=False)
        if parent['dim'] != dim or not data['graph_degree'] < parent['n'] < n or parent['graph_degree'] != data['graph_degree'] or parent.get('reordered') or parent['graph_format'] != 'rover':
            raise ValueError('Parent must be a smaller graph with original IDs, matching dim and degree')
        prefix_hash = hashlib.sha256()
        with base.open('rb') as f:
            remaining = parent['n'] * (dim+1)*4
            while remaining:
                chunk = f.read(min(8*1024*1024, remaining))
                if not chunk:
                    raise ValueError('Truncated parent prefix')
                prefix_hash.update(chunk); remaining -= len(chunk)
        if prefix_hash.hexdigest() != parent['raw_sha256']:
            raise ValueError('Parent rows are not the prefix of the supplied base')
    identity_fields = dict(schema='rover.insert.checkpoint.v1', raw=data['raw_sha256'], n=n, dim=dim,
                           algorithm='rover.stream-insert.v1', rpc_abi=capabilities['rpc_abi'],
                           method=args.method, bits=args.sq_bits if args.method=='sq' else args.pq_bits,
                           residual_bits=args.residual_bits, third_residual=args.third_residual,
                           graph_degree=args.graph_degree, candidates=args.build_candidates, pool=args.build_pool,
                           entry_points=args.entry_points, seed=args.seed, lanes=args.lanes, window=args.source_window,
                           expand_count=args.expand_count, correction=args.correction, split_distance=args.split_distance,
                           backlink_threads=args.backlink_threads,
                           source={k:v['sha256'] for k,v in data['files'].items() if k not in ('graph', 'error4') or (k=='error4' and args.third_residual=='on')})
    identity = hashlib.sha256(json.dumps(identity_fields, sort_keys=True).encode()).hexdigest()
    job_id = uuid.uuid4().hex
    results = (args.results_dir or ROOT/'runs'/job_id).resolve()
    results.mkdir(parents=True, exist_ok=False)
    device = Device(args.serial, args.device_dir)
    job = device.root+'/jobs/'+job_id
    remote_index = job+'/index'
    device.binaries()
    device.mkdir(remote_index)
    native = [device.root+'/bin/Rover', args.command, '--index-dir', remote_index,
              '--n', str(n), '--dim', str(dim), '--method', args.method,
              '--graph-entry',str(data.get('graph_entry',0)), '--graph-degree', str(data['graph_degree']), '--graph-format', data['graph_format'],
              '--layout', data['layout'], '--layout-hash', data['layout_hash'], '--result-dir', job,
              '--metrics', job+'/metrics.csv', '--output', job+'/results.jsonl', '--identity', identity,
              '--checkpoint', remote_index+'/insert.checkpoint']
    for key, item in data['files'].items():
        if key == 'error4' and args.third_residual == 'off':
            continue
        local = directory/item['path']
        remote = remote_index+'/'+Path(item['path']).name
        device.push(local, remote)
        if key in ('graph', 'order_map', 'residual_map', 'l1', 'residual', 'qparams', 'error4', 'bias', 'pq_book', 'pq_codes', 'pq_residual', 'pq_qparams'):
            native += ['--'+key.replace('_','-'), remote]
    if args.method == 'pq':
        native += ['--pq-prefix', remote_index+'/index', '--pq-bits', str(args.pq_bits), '--pq-m', str(args.pq_m)]
    else:
        native += ['--sq-bits', str(args.sq_bits)]
    if construction:
        device.push(base, job+'/base.fvecs')
        native += ['--base', job+'/base.fvecs', '--graph', remote_index+'/graph.rover']
    else:
        device.push(args.query, job+'/query.fvecs')
        native += ['--query', job+'/query.fvecs', '--nq', str(nq)]
        if args.gt:
            native += ['--gt-k', str(gt_k)]
            device.push(args.gt, job+'/groundtruth.ivecs')
            native += ['--gt', job+'/groundtruth.ivecs']
    if args.checkpoint:
        device.push(args.checkpoint, remote_index+'/insert.checkpoint')
    if parent:
        device.push(args.parent_index/parent['files']['graph']['path'], job+'/parent.rover')
        native += ['--parent-graph', job+'/parent.rover', '--parent-n', str(parent['n'])]
    for name in ('residual_bits', 'third_residual', 'correction', 'refine', 'storage', 'search_mode', 'top_k', 'pool',
                 'entry_points', 'lanes', 'source_window', 'first_stage_factor', 'cache_pages', 'refine_cache_pages',
                 'refine_factor', 'seed', 'build_method', 'build_pool', 'build_candidates', 'build_iterations', 'build_threads',
                 'build_refine_candidates','build_distance','rnn_s','rnn_t1','expand_count', 'backlink_threads', 'reverse_candidates', 'checkpoint_every', 'cooldown_every', 'cooldown_ms', 'domain', 'unsigned_pd', 'split_distance'):
        native += ['--'+name.replace('_','-'), str(getattr(args,name))]
    if args.refine_all:
        native.append('--refine-all')
    atomic_json(results/'request.json', dict(arguments=native, identity=identity_fields, index=data))
    env = ['env', 'LD_LIBRARY_PATH='+device.root+'/bin',
           'ADSP_LIBRARY_PATH='+device.root+'/bin;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp']
    success = False
    try:
        with (results/'device.log').open('wb') as log:
            device.shell(env+native, stdout=log, stderr=subprocess.STDOUT)
        success = True
    finally:
        if construction:
            subprocess.run(device.adb+['pull', remote_index+'/insert.checkpoint', str(results/'insert.checkpoint')], check=False)
            if args.cooldown_every:
                subprocess.run(device.adb+['pull', job+'/cooldown.jsonl', str(results/'cooldown.jsonl')], check=False)
    if success and construction:
        graph = directory/'graph.rover'
        if graph.exists():
            raise ValueError('Output graph already exists; use a fresh bundle')
        device.call('pull', remote_index+'/graph.rover', graph)
        read_graph(graph, 'rover', n, data['graph_degree'], dim)
        data['files']['graph'] = record(graph, directory)
        data['state'] = 'complete'
        atomic_json(directory/'index.json', data)
    elif success:
        device.call('pull', job+'/results.jsonl', results/'results.jsonl')
        subprocess.run(device.adb+['pull', job+'/metrics.csv', str(results/'metrics.csv')], check=False)
    print(results)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print('Rover: '+str(error), file=sys.stderr)
        raise SystemExit(1)

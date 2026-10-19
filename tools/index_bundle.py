from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import numpy as np
from generate_final_refine import SqSource, sha, ref
from pq_refine_artifacts import PqSource, generate_second
import encode_l1_residual as sq
import build_pure_pq_index as pq
import build_pure_pq_residual as residual

SCHEMA = 'rover.index.v1'
INTRINSIC = ('method', 'sq_bits', 'pq_bits', 'pq_m', 'residual_bits', 'graph_degree')


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def record(path, root):
    path, root = Path(path), Path(root)
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path))


def vector_shape(path):
    n, dim = sq.read_fvecs_header(Path(path))
    if n < 1 or not 0 < dim <= 1024:
        raise ValueError('Rover requires nonempty fvecs with dimension 1..1024')
    return n, dim


def read_graph(path, kind, n, degree, dim):
    path = Path(path)
    if kind == 'rover':
        if path.stat().st_size != 16 + n*degree*4:
            raise ValueError('Rover graph size mismatch')
        with path.open('rb') as f:
            if f.read(16) != struct.pack('<4i', 0x35314252, n, dim, degree):
                raise ValueError('Rover graph header mismatch')
        graph = np.memmap(path, mode='r', offset=16, dtype='<i4', shape=(n, degree))
    elif kind == 'dense':
        if path.stat().st_size != n * degree * 4:
            raise ValueError('Dense graph size mismatch')
        graph = np.memmap(path, mode='r', dtype='<i4', shape=(n, degree))
    elif kind == 'nndescent':
        with path.open('rb') as f:
            header = struct.unpack('<3i', f.read(12))
        if header != (n, degree, 0) or path.stat().st_size != 12 + n * degree * 8:
            raise ValueError('NNDescent graph shape mismatch')
        rows = np.memmap(path, mode='r', offset=12, dtype='u1', shape=(n, degree*8))
        graph = rows[:, :degree*4].copy().view('<i4').reshape(n, degree)
    elif kind == 'nsg':
        graph = np.full((n, degree), -1, dtype='<i4')
        with path.open('rb') as f:
            width, entry = struct.unpack('<2I', f.read(8))
            if not width or entry >= n:
                raise ValueError('NSG header mismatch')
            for i in range(n):
                k = struct.unpack('<I', f.read(4))[0]
                if k > width:
                    raise ValueError('NSG row exceeds declared width')
                if k > degree:
                    raise ValueError('NSG row exceeds graph-degree; provide a graph within the requested width')
                row = np.frombuffer(f.read(k*4), dtype='<i4')
                if len(row) != k:
                    raise ValueError('Truncated NSG graph')
                graph[i, :min(k, degree)] = row[:degree]
            if f.read(1):
                raise ValueError('Trailing NSG graph rows')
    else:
        raise ValueError('Unknown graph format: '+kind)
    for start in range(0, n, 8192):
        rows = graph[start:start+8192]
        if np.any(rows < -1) or np.any(rows >= n):
            raise ValueError('Out-of-range graph neighbor')
        if np.any(rows == np.arange(start, start+len(rows))[:, None]):
            raise ValueError('Graph contains a self edge')
        ordered=np.sort(rows, axis=1)
        if np.any((ordered[:, 1:] >= 0) & (ordered[:, 1:] == ordered[:, :-1])):
            raise ValueError('Graph contains duplicate edges')
    return graph


def write_graph(path, graph, dim):
    n, degree = graph.shape
    with Path(path).open('xb') as f:
        f.write(struct.pack('<4i', 0x35314252, n, dim, degree))
        for start in range(0, n, 8192):
            f.write(np.asarray(graph[start:start+8192], dtype='<i4').tobytes())
        f.flush()
        os.fsync(f.fileno())


def reorder(base, graph, directory, args):
    import faiss
    n, dim = vector_shape(base)
    work = directory / 'preparation'
    work.mkdir()
    raw = np.memmap(base, mode='r', dtype='<f4', shape=(n, dim+1))
    clusters = args.num_clusters or max(1, int(np.sqrt(n)))
    if not 1 <= clusters <= n:
        raise ValueError('num-clusters must be in 1..N')
    trainer = faiss.Kmeans(dim, clusters, niter=args.cluster_iters, seed=args.seed, verbose=False)
    samples = np.linspace(0, n-1, min(n, max(100000, clusters*40)), dtype=np.int64)
    trainer.train(np.ascontiguousarray(raw[samples, 1:]))
    labels = work / 'clusters.txt'
    with labels.open('x') as f:
        for begin in range(0, n, 8192):
            _, ids = trainer.index.search(np.ascontiguousarray(raw[begin:begin+8192, 1:]), 1)
            np.savetxt(f, ids[:, 0], fmt='%d')
    nsg = work / 'layout.graph'
    with nsg.open('xb') as f:
        f.write(struct.pack('<2I', graph.shape[1], 0))
        for row in graph:
            valid = np.asarray(row[row >= 0], dtype='<u4')
            f.write(struct.pack('<I', len(valid)))
            f.write(valid.tobytes())
    order_path = directory / 'order.txt'
    layout = Path(__file__).resolve().parents[1] / 'dist/host/rover-layout'
    if not layout.is_file():
        raise ValueError('Compile the host layout tool first: python3 scripts/build.py --target host')
    vpb = 4096 // ((dim*args.residual_bits+7)//8)
    subprocess.run([str(layout), str(nsg), str(labels), str(order_path), str(clusters), str(vpb),
                    '60', '64', str(args.host_threads), str(int(args.cn == 'on')), str(int(args.mnn == 'on'))], check=True)
    order = np.loadtxt(order_path, dtype='<i4', ndmin=1)
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError('Layout output is not a permutation')
    inverse = np.empty(n, dtype='<i4')
    inverse[order] = np.arange(n, dtype='<i4')
    mapped = np.empty_like(graph)
    reordered = work / 'base.fvecs'
    with reordered.open('xb') as f:
        for begin in range(0, n, 8192):
            ids = order[begin:begin+8192]
            f.write(np.asarray(raw[ids], dtype='<f4').tobytes())
            neighbors = graph[ids]
            mapped[begin:begin+len(ids)] = np.where(neighbors < 0, -1, inverse[np.maximum(neighbors, 0)])
    if args.graph_entry is not None and args.graph_entry>=0:
        args.graph_entry=int(inverse[args.graph_entry])
    return reordered, mapped, order_path


def resolve_codec(args, dim, metadata=None):
    metadata = metadata or {}
    defaults = dict(method='sq', sq_bits=4, pq_bits=8, residual_bits=4, graph_degree=16)
    for name, default in defaults.items():
        value = getattr(args, name)
        saved = metadata.get(name, default)
        if metadata and name in metadata and value is not None and value != saved:
            raise ValueError(f'--{name.replace("_", "-")} conflicts with index metadata; use --reencode and a new output directory')
        setattr(args, name, saved if value is None else value)
    if args.method == 'pq':
        selected = sum(getattr(args, key) is not None for key in ('pq_m', 'pq_subdim', 'pq_compression'))
        if selected > 1:
            raise ValueError('Use exactly one of pq-m, pq-subdim, pq-compression')
        m = args.pq_m
        if args.pq_subdim is not None:
            if args.pq_subdim <= 0 or dim % args.pq_subdim:
                raise ValueError('PQ subdim must divide dimension')
            m = dim // args.pq_subdim
        if args.pq_compression is not None:
            denominator = args.pq_compression * args.pq_bits
            if denominator <= 0 or dim*32 % denominator:
                raise ValueError('PQ compression requires an integral M')
            m = dim*32 // denominator
        if m is None:
            m = metadata.get('pq_m', dim//16 if dim % 16 == 0 else dim)
        if m <= 0 or dim % m or (metadata and m != metadata['pq_m']):
            raise ValueError('Invalid or conflicting PQ M')
        args.pq_m = m
    if not 0 < args.graph_degree < (metadata.get('n', 2**31-1)):
        raise ValueError('Invalid graph degree')


def validate(directory, third=True, require_graph=True):
    directory = Path(directory).resolve()
    data = json.loads((directory/'index.json').read_text())
    if data.get('schema') != SCHEMA:
        raise ValueError('Unsupported Rover index schema')
    if data.get('method') not in ('sq', 'pq') or not 0 < data['dim'] <= 1024 or not 0 < data['graph_degree'] < data['n'] <= 2147483647:
        raise ValueError('Invalid index method or dimensions')
    if data['residual_bits'] not in (4, 8) or data.get('sq_bits' if data['method']=='sq' else 'pq_bits') not in (4, 8):
        raise ValueError('Index bits must be 4 or 8')
    required = ['l1', 'residual', 'qparams'] if data['method'] == 'sq' else ['pq_book', 'pq_codes', 'pq_residual', 'pq_qparams', 'pq_manifest', 'pq_residual_manifest']
    if require_graph:
        required.append('graph')
    if third:
        required.append('error4')
    if data.get('reordered'):
        required.append('order_map')
    if data.get('residual_layout','a') != 'a':
        required.append('residual_map')
    for key in required:
        if key not in data['files']:
            raise ValueError('Index missing '+key)
    remote_names=set()
    for key, item in data['files'].items():
        if key == 'error4' and not third:
            continue
        path = (directory/item['path']).resolve()
        if Path(item['path']).is_absolute() or not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('Invalid index member '+key)
        if path.name in remote_names or path.name == 'insert.checkpoint':
            raise ValueError('Colliding index member names')
        remote_names.add(path.name)
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Index member changed: '+key)
    if data['method'] == 'pq':
        suffix=f'.pqres{data["residual_bits"]}'
        names=dict(pq_book='index.pqbook', pq_codes='index.pqcodes', pq_manifest='index.pq.json',
                   pq_residual='index'+suffix, pq_qparams='index'+suffix+'.qparams', pq_residual_manifest='index'+suffix+'.json')
        if any(Path(data['files'][key]['path']).name != name for key, name in names.items()):
            raise ValueError('PQ bundle filenames must use the index prefix')
    return data


def prepare(args, construction=False):
    directory = Path(args.output_index_dir or args.index_dir).resolve()
    if (directory/'index.json').exists() and not args.reencode and args.reorder == 'off':
        data = validate(directory, third=args.third_residual == 'on', require_graph=not construction)
        if args.residual_layout != 'a' and args.residual_layout != data.get('residual_layout','a'):
            raise ValueError('Residual layout conflicts with existing index; reencode into a new directory')
        resolve_codec(args, data['dim'], data)
        return directory, data, Path(args.base).resolve() if args.base else None
    if not args.base:
        raise ValueError('Preparing or reencoding an index requires --base')
    base = Path(args.base).resolve()
    n, dim = vector_shape(base)
    resolve_codec(args, dim)
    if args.graph_degree >= n:
        raise ValueError('graph-degree must be smaller than N')
    if args.graph_entry is not None and not -1 <= args.graph_entry < n:
        raise ValueError('graph-entry must be -1 or a valid input row')
    if args.split_distance == 'on' and (args.method != 'sq' or args.sq_bits != 4 or args.correction != 'online'):
        raise ValueError('Split distance requires SQ4 and online correction')
    if args.command in ('build-index', 'insert', 'resume'):
        if args.build_method == 'rnn-descent':
            if args.rnn_s > args.graph_degree:
                raise ValueError('rnn-s must not exceed graph-degree')
        elif args.build_candidates < args.graph_degree:
            raise ValueError('build-candidates must cover graph-degree')
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Preparation requires a new, empty output directory')
    directory.mkdir(parents=True, exist_ok=True)
    graph = None
    order_path = None
    if args.graph:
        graph = read_graph(args.graph, args.graph_format, n, args.graph_degree, dim)
    if not construction and graph is None:
        raise ValueError('Importing a search index requires --graph and --base')
    if args.graph and args.graph_format=='nsg' and args.graph_entry is None:
        with Path(args.graph).open('rb') as f:
            args.graph_entry=struct.unpack('<2I',f.read(8))[1]
    if args.reorder == 'on':
        if graph is None:
            raise ValueError('Graph reordering requires an existing graph')
        base, graph, order_path = reorder(base, graph, directory, args)
    data = dict(schema=SCHEMA, n=n, dim=dim, graph_degree=args.graph_degree, method=args.method,
                residual_bits=args.residual_bits, third_residual=args.third_residual == 'on',
                reordered=bool(order_path), layout='original', layout_hash=hashlib.sha256(b'original').hexdigest(),
                graph_format='rover', graph_entry=args.graph_entry if args.graph_entry is not None else 0, files={}, raw_sha256=sha(base))
    files = data['files']
    if graph is not None:
        write_graph(directory/'graph.rover', graph, dim)
        files['graph'] = record(directory/'graph.rover', directory)
    if order_path:
        files['order_map'] = record(order_path, directory)
    data['residual_layout']=args.residual_layout
    if args.residual_layout != 'a' and (args.method!='sq' or graph is None):
        raise ValueError('Residual B/C/D layouts require SQ and an existing graph')
    if args.method == 'sq':
        data['sq_bits'] = args.sq_bits
        paths = [directory/'l1.bin', directory/'residual.bin', directory/'qparams.bin']
        sq.encode(base, *paths, args.sq_bits, args.residual_bits, args.chunk_vectors)
        storage = None
        if args.residual_layout != 'a':
            tool = Path(__file__).resolve().parents[1]/'dist/host/rover-residual-layout'
            storage_path = directory/'residual-map.txt'
            cb = (dim*args.residual_bits+7)//8
            vpb = 4096//cb
            subprocess.run([str(tool),str(directory/'graph.rover'),str('abcd'.index(args.residual_layout)),str(vpb),str(storage_path)],check=True)
            storage = np.loadtxt(storage_path,dtype='<i4',ndmin=1)
            if not np.array_equal(np.sort(storage),np.arange(n)):
                raise ValueError('Invalid residual permutation')
            old = np.memmap(paths[1],mode='r',dtype='u1').reshape(-1,4096)
            temporary = directory/'residual.mapped.tmp'
            with temporary.open('xb') as f:
                f.truncate(paths[1].stat().st_size)
            new = np.memmap(temporary,mode='r+',dtype='u1').reshape(-1,4096)
            for begin in range(0,n,8192):
                ids=np.arange(begin,min(n,begin+8192));slots=storage[ids]
                new[(slots//vpb)[:,None],((slots%vpb)*cb)[:,None]+np.arange(cb)] = old[(ids//vpb)[:,None],((ids%vpb)*cb)[:,None]+np.arange(cb)]
            new.flush();del new,old
            os.replace(temporary,paths[1])
            files['residual_map']=record(storage_path,directory)
        for key, path in zip(('l1', 'residual', 'qparams'), paths):
            files[key] = record(path, directory)
        source_manifest = dict(N=n, dim=dim, layout='original', order=None, **{key: files[key] for key in ('l1', 'residual', 'qparams')})
        if order_path:
            source_manifest['external_order']=files['order_map']
        if storage is not None:
            source_manifest['residual_map']=files['residual_map']
        atomic_json(directory/'codec.json', source_manifest)
        if args.third_residual == 'on':
            generate_second(MappedSqSource(directory/'codec.json', base), directory/'error4.refine', paths)
        if args.split_distance == 'on':
            import generate_sqmd_correction_bias
            generate_sqmd_correction_bias.generate(directory/'codec.json', directory/'correction_bias.f32', args.chunk_vectors)
            files['bias'] = record(directory/'correction_bias.f32', directory)
    else:
        data.update(pq_bits=args.pq_bits, pq_m=args.pq_m, train_samples=args.pq_train_samples,
                    train_iters=args.pq_train_iters, seed=args.seed)
        prefix = directory/'index'
        pq.build(argparse.Namespace(base=base, out_prefix=prefix, num_vectors=n, dim=dim, m=args.pq_m,
                  code_bits=args.pq_bits, ks=None, train_max=args.pq_train_samples, kmeans_iters=args.pq_train_iters,
                  chunk_vectors=args.chunk_vectors, memory_budget_mb=args.memory_budget_mb, layout_id='original',
                  order_map=None, engine='faiss', threads=args.host_threads, seed=args.seed, force=False))
        residual.build(argparse.Namespace(prefix=prefix, base=base, num_vectors=n, residual_bits=args.residual_bits,
                       chunk_vectors=args.chunk_vectors, order_map=None, force=False))
        suffix = f'.pqres{args.residual_bits}'
        for key, ext in (('pq_book', '.pqbook'), ('pq_codes', '.pqcodes'), ('pq_manifest', '.pq.json'),
                         ('pq_residual', suffix), ('pq_qparams', suffix+'.qparams'), ('pq_residual_manifest', suffix+'.json')):
            files[key] = record(Path(str(prefix)+ext), directory)
        if args.third_residual == 'on':
            paths = [directory/files[key]['path'] for key in ('pq_codes', 'pq_residual', 'pq_qparams')]
            source=PqSource(prefix,base,args.residual_bits)
            if order_path:
                original=np.loadtxt(order_path,dtype='<i4',ndmin=1)
                source.mapping_sha=hashlib.sha256(original.tobytes()+np.arange(n,dtype='<i4').tobytes()).digest()
            generate_second(source,directory/'error4.refine',paths,directory/files['pq_book']['path'])
    if args.third_residual == 'on':
        files['error4'] = record(directory/'error4.refine', directory)
    atomic_json(directory/'index.json', data)
    return directory, data, base


class MappedSqSource(SqSource):
    def __init__(self, manifest, base):
        super().__init__(manifest, base)
        if self.m.get('residual_map'):
            self.storage=np.loadtxt(self.manifest_path.parent/self.m['residual_map']['path'],dtype='<i4',ndmin=1)
            if not np.array_equal(np.sort(self.storage),np.arange(self.n)):
                raise ValueError('Invalid residual map')
        if self.storage is not None or self.m.get('external_order'):
            original=(np.loadtxt(self.manifest_path.parent/self.m['external_order']['path'],dtype='<i4',ndmin=1)
                      if self.m.get('external_order') else np.arange(self.n,dtype='<i4'))
            slots=self.storage if self.storage is not None else np.arange(self.n,dtype='<i4')
            self.mapping_sha=hashlib.sha256(original.tobytes()+slots.tobytes()).digest()

    def reconstruct(self, ids):
        from generate_final_refine import unpack
        ids=np.asarray(ids,dtype=np.int64)
        slots=ids if self.storage is None else self.storage[ids]
        packed=self.res[(slots//self.vpb)[:,None],((slots%self.vpb)*self.rc)[:,None]+np.arange(self.rc)]
        qp=self.qp[ids]
        a=unpack(self.l1[ids],self.lb,self.dim)*self.scale+self.zero
        return (a+(unpack(packed,self.rb,self.dim)*qp[:,0:1]+qp[:,1:2])).astype('<f4')

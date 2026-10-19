#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
import numpy as np

HEADER = struct.Struct('<8s8I2Q32s96s32s32s8s')
L1_HEADER = struct.Struct('<IIiiiff4s')
QP_HEADER = struct.Struct('<IIiiff8s')
BLOCK = 4096
NUMERIC = 'f32-separate-mul-add-v1;f64-l2;half-away-from-zero;low-nibble-first'


def sha(path, offset=0):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        f.seek(offset)
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ref(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def unpack(codes, bits, dim):
    if bits == 8:
        return np.asarray(codes, dtype=np.float32)
    result = np.empty((len(codes), dim), dtype=np.float32)
    result[:, 0::2] = codes[:, :(dim+1)//2] & 15
    result[:, 1::2] = codes[:, :dim//2] >> 4
    return result


def quantize_error(error):
    zero = error.min(axis=1)
    scale = (np.maximum(error.max(axis=1)-zero, np.float32(1e-12))/np.float32(15)).astype('<f4')
    normalized = (error-zero[:, None])*(np.float32(1)/scale[:, None])
    codes = np.clip(np.floor(normalized + np.float32(0.5)), 0, 15).astype('u1')
    packed = np.zeros((len(codes), (codes.shape[1]+1)//2), dtype='u1')
    packed[:] = codes[:, 0::2]
    packed[:, :codes.shape[1]//2] |= codes[:, 1::2] << 4
    return packed, np.column_stack((scale, zero)).astype('<f4')


class SqSource:
    def __init__(self, manifest_path, base_path, verify=True):
        self.manifest_path = Path(manifest_path).resolve()
        self.m = json.loads(self.manifest_path.read_text())
        self.n, self.dim = int(self.m['N']), int(self.m['dim'])
        if self.n <= 0 or not 0 < self.dim <= 8192:
            raise ValueError('invalid dimensions')
        self.paths = {}
        for key in ('l1', 'residual', 'qparams'):
            r = self.m[key]
            p = Path(r['path'])
            p = p if p.is_absolute() else self.manifest_path.parent/p
            if p.stat().st_size != r['bytes'] or (verify and sha(p) != r['sha256']):
                raise ValueError('base representation mismatch: '+key)
            self.paths[key] = p
        with self.paths['l1'].open('rb') as f:
            lh = L1_HEADER.unpack(f.read(32))
        with self.paths['qparams'].open('rb') as f:
            rh = QP_HEADER.unpack(f.read(32))
        if (lh[:4] != (0x314c4d53, 1, self.n, self.dim) or
                rh[:3] != (0x51504d53, 1, self.n) or lh[4] not in (4,8) or
                rh[3] not in (4,8) or lh[5:7] != rh[4:6] or
                not np.isfinite(lh[5:7]).all() or lh[5] <= 0):
            raise ValueError('invalid source headers')
        self.lb, self.rb, self.scale, self.zero = lh[4], rh[3], np.float32(lh[5]), np.float32(lh[6])
        lc, rc = (self.dim*self.lb+7)//8, (self.dim*self.rb+7)//8
        self.rc, self.vpb = rc, BLOCK//rc
        if not self.vpb or self.paths['l1'].stat().st_size != 32+self.n*lc or self.paths['qparams'].stat().st_size != 32+8*self.n or self.paths['residual'].stat().st_size != ((self.n+self.vpb-1)//self.vpb)*BLOCK:
            raise ValueError('source payload size mismatch')
        self.l1 = np.memmap(self.paths['l1'],mode='r',dtype='u1',offset=32,shape=(self.n,lc))
        self.res = np.memmap(self.paths['residual'],mode='r',dtype='u1').reshape(-1,BLOCK)
        self.qp = np.memmap(self.paths['qparams'],mode='r',dtype='<f4',offset=32,shape=(self.n,2))
        self.base_path = Path(base_path).resolve()
        if self.base_path.stat().st_size != self.n*(self.dim+1)*4:
            raise ValueError('raw fvecs size mismatch')
        self.raw = np.memmap(self.base_path,mode='r',dtype='<f4',shape=(self.n,self.dim+1))
        self.original = None
        if self.m['layout'] == 'gorder':
            r=self.m['order'];p=Path(r['path'])
            p=p if p.is_absolute() else self.manifest_path.parent/p
            if sha(p)!=r['sha256'] or p.stat().st_size!=4*self.n:
                raise ValueError('invalid order map')
            self.original=np.fromfile(p,dtype='<i4')
            if not np.array_equal(np.sort(self.original),np.arange(self.n)):
                raise ValueError('order is not a permutation')
        elif self.m['layout'] != 'original':
            raise ValueError('unsupported graph layout')
        self.storage = None
        self.mapping_sha = hashlib.sha256(b'' if self.original is None else self.original.tobytes()+np.arange(self.n,dtype='<i4').tobytes()).digest()

    def reconstruct(self, ids):
        ids=np.asarray(ids,dtype=np.int64)
        packed=self.res[(ids//self.vpb)[:,None],((ids%self.vpb)*self.rc)[:,None]+np.arange(self.rc)]
        qp=np.asarray(self.qp[ids])
        if not np.isfinite(qp).all() or (qp[:,0]<0).any():
            raise ValueError('invalid quantization parameters')
        a=unpack(self.l1[ids],self.lb,self.dim)*self.scale+self.zero
        r=unpack(packed,self.rb,self.dim)*qp[:,0:1]+qp[:,1:2]
        return (a+r).astype('<f4')

    def raw_vectors(self, ids):
        ids=np.asarray(ids,dtype=np.int64)
        rows=self.raw[ids if self.original is None else self.original[ids]]
        if not (rows[:,0].view('<i4') == self.dim).all() or not np.isfinite(rows[:,1:]).all():
            raise ValueError('invalid raw fvecs record')
        return rows[:,1:]


def generate(manifest_path, base_path, output, chunk=4096):
    output=Path(output).resolve(); marker=output.with_suffix('.json')
    if output.exists() or marker.exists():
        raise FileExistsError('preserve existing artifact: '+str(output))
    if chunk<=0: raise ValueError('chunk must be positive')
    s=SqSource(manifest_path,base_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_name(output.name+'.tmp')
    cb=(s.dim+1)//2;vpb=BLOCK//cb;blocks=(s.n+vpb-1)//vpb
    qb=8*s.n; code_offset=HEADER.size+qb
    try:
        with temp.open('xb') as f: f.truncate(code_offset+blocks*BLOCK)
        params=np.memmap(temp,mode='r+',dtype='<f4',offset=HEADER.size,shape=(s.n,2))
        codes=np.memmap(temp,mode='r+',dtype='u1',offset=code_offset,shape=(blocks,BLOCK))
        for begin in range(0,s.n,chunk):
            ids=np.arange(begin,min(begin+chunk,s.n))
            xhat=s.reconstruct(ids);error=(s.raw_vectors(ids)-xhat).astype('<f4')
            if not np.isfinite(error).all():raise ValueError('nonfinite reconstruction error')
            packed,qp=quantize_error(error)
            params[ids]=qp
            codes[(ids//vpb)[:,None],((ids%vpb)*cb)[:,None]+np.arange(cb)]=packed
        params.flush();codes.flush();del params,codes
        payload=bytes.fromhex(sha(temp,HEADER.size))
        base_hashes=b''.join(bytes.fromhex(s.m[k]['sha256']) for k in ('l1','residual','qparams'))
        raw_ref=ref(s.base_path)
        h=HEADER.pack(b'SQERR4V1',1,s.n,s.dim,s.lb,s.rb,BLOCK,1,1,qb,blocks*BLOCK,payload,base_hashes,bytes.fromhex(raw_ref['sha256']),s.mapping_sha,b'\0'*8)
        with temp.open('r+b') as f:
            f.write(h);f.flush();os.fsync(f.fileno())
        os.replace(temp,output)
        doc=dict(schema='phase15.error4.v1',N=s.n,dim=s.dim,l1_bits=s.lb,residual_bits=s.rb,
            numeric=NUMERIC,layout=s.m['layout'],mapping_sha256=s.mapping_sha.hex(),
            sources={k:s.m[k] for k in ('l1','residual','qparams','order')},raw=raw_ref,
            artifact=ref(output),header_bytes=HEADER.size,qparams_bytes=qb,code_blocks=blocks,
            block_bytes=BLOCK,vectors_per_block=vpb,code_bytes=cb,
            source_manifest=ref(s.manifest_path))
        tmp=marker.with_name(marker.name+'.tmp');tmp.write_text(json.dumps(doc,indent=2)+'\n');os.replace(tmp,marker)
        return doc
    except BaseException:
        temp.unlink(missing_ok=True)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=(
        'Phase15 ERROR4 artifact. No query/GT inputs enter encoding.\n'
        '\n'
        'Numeric v1: separate float32 multiply/add reconstruction, half-away-from-zero\n'
        'min/max error quantization. Row order and maps are bound by SHA256. Header is\n'
        '256 bytes, followed by resident float32 scale/zero and 4096-byte code blocks.\n'
        'The JSON manifest is published last and is the completion marker.\n'
    ))
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--chunk',type=int,default=4096)
    a=p.parse_args();print(json.dumps(generate(a.manifest,a.base,a.output,a.chunk),indent=2))

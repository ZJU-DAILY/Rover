import hashlib
import os
from pathlib import Path
import numpy as np
import build_pure_pq_index as pq
import build_pure_pq_residual as residual
from generate_final_refine import HEADER, unpack, quantize_error, sha, ref

class PqSource:
    def __init__(self,prefix,base,residual_bits=4):
        self.prefix=Path(prefix);self.meta=pq.validate_prefix(self.prefix)
        self.resmeta=residual.validate(self.prefix,residual_bits)
        for key in ('index_uuid','dataset_fingerprint','layout_fingerprint'):
            if self.meta[key]!=self.resmeta[key]: raise ValueError('PQ/residual identity mismatch')
        self.n,self.dim=self.meta['num_vectors'],self.meta['dim']
        self.lb,self.rb=pq.code_bits_for_ks(self.meta['ks']),residual_bits
        self.book=np.memmap(str(prefix)+'.pqbook',mode='r',offset=256,dtype='<f4',
            shape=(self.meta['m'],self.meta['ks'],self.meta['subdim']))
        self.codes=np.memmap(str(prefix)+'.pqcodes',mode='r',offset=256,dtype='u1',shape=(self.n,self.meta['code_bytes']))
        self.qp=np.memmap(str(prefix)+f'.pqres{self.rb}.qparams',mode='r',offset=256,dtype='<f4',shape=(self.n,2))
        self.res=np.memmap(str(prefix)+f'.pqres{self.rb}',mode='r',offset=4096,dtype='u1').reshape(-1,4096)
        self.rc=(self.dim*self.rb+7)//8;self.vpb=4096//self.rc;self.storage=None
        self.mapping_sha=hashlib.sha256(b'').digest()
        self.base_path=Path(base);self.raw=np.memmap(base,mode='r',dtype='<f4',shape=(self.n,self.dim+1))

    def reconstruct(self,ids):
        ids=np.asarray(ids,dtype=np.int64)
        centroid=residual.reconstruct_batch(self.codes[ids],self.book)
        packed=self.res[(ids//self.vpb)[:,None],((ids%self.vpb)*self.rc)[:,None]+np.arange(self.rc)]
        params=self.qp[ids]
        product=unpack(packed,self.rb,self.dim)*params[:,0:1]
        return ((centroid+product)+params[:,1:2]).astype('<f4')

    def raw_vectors(self,ids):
        rows=self.raw[ids]
        if not (rows[:,0].view('<i4')==self.dim).all() or not np.isfinite(rows[:,1:]).all():
            raise ValueError('invalid raw vector')
        return rows[:,1:]


def generate_second(source, output, base_paths, pq_book=None):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    n,dim=source.n,source.dim;cb=(dim+1)//2;vpb=4096//cb;blocks=(n+vpb-1)//vpb
    temp=output.with_suffix('.tmp')
    with temp.open('xb') as stream: stream.truncate(256+8*n+4096*blocks)
    params=np.memmap(temp,mode='r+',offset=256,dtype='<f4',shape=(n,2))
    codes=np.memmap(temp,mode='r+',offset=256+8*n,dtype='u1',shape=(blocks,4096))
    for begin in range(0,n,8192):
        ids=np.arange(begin,min(begin+8192,n))
        error=source.raw_vectors(ids)-source.reconstruct(ids)
        if not np.isfinite(error).all(): raise ValueError('nonfinite encoding error')
        packed,qp=quantize_error(error)
        slots=ids if source.storage is None else source.storage[ids]
        params[slots]=qp
        codes[(slots//vpb)[:,None],((slots%vpb)*cb)[:,None]+np.arange(cb)]=packed
    params.flush();codes.flush();del params,codes
    header=HEADER.pack(b'PQERR4V1' if pq_book else b'SQERR4V1',1,n,dim,source.lb,source.rb,4096,1,1,
        8*n,blocks*4096,bytes.fromhex(sha(temp,256)),b''.join(bytes.fromhex(sha(p)) for p in base_paths),
        bytes.fromhex(sha(pq_book or source.base_path)),source.mapping_sha,b'\0'*8)
    with temp.open('r+b') as stream: stream.write(header);stream.flush();os.fsync(stream.fileno())
    temp.replace(output)
    return ref(output)

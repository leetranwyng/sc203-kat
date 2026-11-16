"""Read one checkpoint from a public ZIP without storing the multi-checkpoint archive."""
import io
import re
import zipfile

class HTTPRangeFile(io.RawIOBase):
    def __init__(self,url,token=None,block_size=8*2**20):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        self.session=requests.Session()
        self.session.mount('https://',HTTPAdapter(max_retries=Retry(total=5,backoff_factor=1,
            status_forcelist=[429,500,502,503,504],allowed_methods=['GET'])))
        if token:self.session.headers['Authorization']='Bearer '+token
        self.url=url;self.pos=0;self.block_size=block_size;self.cached_start=-1;self.cache=b''
        with self.session.get(url,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'},
                              timeout=120,stream=True) as response:
            response.raise_for_status()
            if response.status_code!=206:raise OSError('Server must support HTTP Range for OneRef ZIP')
            match=re.fullmatch(r'bytes 0-0/(\d+)',response.headers.get('Content-Range',''))
            if not match:raise OSError('Invalid Content-Range')
            self.size=int(match[1])
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('Negative ZIP seek')
        return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        chunks=[]
        while n>0:
            if not self.cached_start<=self.pos<self.cached_start+len(self.cache):
                start=self.pos;end=min(self.size-1,start+max(self.block_size,min(n,32*2**20))-1)
                with self.session.get(self.url,headers={'Range':f'bytes={start}-{end}',
                        'Accept-Encoding':'identity'},timeout=120,stream=True) as response:
                    response.raise_for_status();expected=f'bytes {start}-{end}/{self.size}'
                    if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                        raise OSError('Server returned the wrong byte range')
                    content=response.content
                if len(content)!=end-start+1:raise OSError('Truncated remote ZIP block')
                self.cached_start=start;self.cache=content
            offset=self.pos-self.cached_start;take=min(n,len(self.cache)-offset)
            chunks.append(self.cache[offset:offset+take]);self.pos+=take;n-=take
        return b''.join(chunks)
    def close(self):
        self.session.close();super().close()

def extract_mixup(url,target,token=None):
    with HTTPRangeFile(url,token) as remote,zipfile.ZipFile(remote) as archive:
        matches=[n for n in archive.namelist() if n.endswith('mixup/best_checkpoint.pth')]
        if len(matches)!=1:raise ValueError('Expected one OneRef mixup checkpoint: '+str(archive.namelist()))
        item=archive.getinfo(matches[0]);tmp=target.with_suffix('.tmp')
        print('OneRef: extracting only mixup checkpoint',item.file_size,'bytes',flush=True)
        try:
            with archive.open(item) as src,tmp.open('wb') as dst:
                while True:
                    block=src.read(8*2**20)
                    if not block:break
                    dst.write(block)
            if tmp.stat().st_size!=item.file_size:raise OSError('Truncated checkpoint')
            tmp.replace(target)
        except BaseException:
            tmp.unlink(missing_ok=True);raise

"""Recover the file index of a Kaggle ZIP whose central directory was truncated."""
import collections
import json
import struct
import sys
from pathlib import Path
from zipfile import ZipFile, ZipInfo

HEADER = struct.Struct('<IHHHHHIIIHH')
LOCAL = 0x04034B50
CENTRAL = 0x02014B50


def local_entries(path):
    end = path.stat().st_size
    infos = []
    with path.open('rb') as stream:
        while stream.tell() + HEADER.size <= end:
            offset = stream.tell()
            tag, version, flags, method, time, date, crc, packed, size, namesz, extrasz = HEADER.unpack(stream.read(HEADER.size))
            if tag == CENTRAL:
                return infos, offset
            if tag != LOCAL or flags & 8:
                raise ValueError(f'Unexpected local ZIP header at {offset}: {tag:x}, flags={flags}')
            name = stream.read(namesz).decode('utf-8' if flags & 2048 else 'cp437')
            extra = stream.read(extrasz)
            if 0xFFFFFFFF in (size, packed):
                cursor = 0
                while cursor + 4 <= len(extra):
                    kind, length = struct.unpack_from('<HH',extra,cursor)
                    cursor += 4
                    if kind == 1:
                        fields = iter(struct.unpack_from('<'+'Q'*(length//8),extra,cursor))
                        size = next(fields) if size == 0xFFFFFFFF else size
                        packed = next(fields) if packed == 0xFFFFFFFF else packed
                        break
                    cursor += length
            if stream.tell()+packed > end:
                return infos, offset
            timestamp = (1980+(date>>9),(date>>5)&15,date&31,time>>11,(time>>5)&63,(time&31)*2)
            info = ZipInfo(name,timestamp)
            info.header_offset,info.compress_type,info.flag_bits = offset,method,flags
            info.CRC,info.compress_size,info.file_size = crc,packed,size
            info.extract_version = version
            infos.append(info)
            stream.seek(packed,1)
    return infos, stream.tell()


def repair(partial, target):
    infos, offset = local_entries(partial)
    assert len(infos) > 100
    counts = collections.Counter(info.filename.split('/')[-1] for info in infos)
    print('Recovered members:',len(infos),'central directory begins:',offset,
          'input images:',counts['input.png'],'orientation records:',counts['orientation_angles_00.json'],flush=True)
    suffix = target.with_suffix('.truncated-central.bin')
    with partial.open('r+b') as stream:
        stream.seek(offset)
        suffix.write_bytes(stream.read())
        stream.truncate(offset)
    with ZipFile(partial,'a',allowZip64=True) as archive:
        archive.filelist = infos
        archive.NameToInfo = {item.filename:item for item in infos}
        archive._didModify = True
    with ZipFile(partial) as archive:
        for info in archive.infolist():
            if info.filename.endswith(('predictions_apc_vis.jsonl','orientation_angles_00.json','summary.json')):
                archive.read(info)
        assert len(archive.infolist()) == len(infos)
    partial.replace(target)
    target.with_suffix('.recovery.json').write_text(json.dumps(
        {'members':len(infos),'original_central_offset':offset,'truncated_index':suffix.name,
         'orientation_rows':counts['orientation_angles_00.json']},indent=2))


if __name__ == '__main__':
    repair(Path(sys.argv[1]), Path(sys.argv[2]))

#!/usr/bin/env python3
"""Inventory placeholders in the active LaTeX input graph, ignoring backups.

This is a manuscript audit, not an experiment or a replacement for PDF review.
Red numeric drafting values are deliberately separate from literal TBD fields.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def uncomment(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)


def braced_argument(text, command):
    match=re.search(re.escape(command)+r'\s*\{', text)
    if not match:
        return None
    start=match.end()
    depth=1
    for i in range(start,len(text)):
        if i and text[i-1]=='\\':
            continue
        if text[i]=='{': depth+=1
        if text[i]=='}': depth-=1
        if not depth: return text[start:i]
    raise ValueError('unbalanced argument: '+command)


def audit(root):
    root=root.resolve()
    sources=[]
    chunks=[]
    def visit(relative,stack=()):
        path=root/relative
        if path.suffix!='.tex': path=path.with_suffix('.tex')
        path=path.resolve()
        if not path.is_relative_to(root) or path in stack:
            raise ValueError('source input escapes root or is cyclic')
        raw=path.read_bytes()
        text=uncomment(raw.decode('utf-8-sig'))
        name=path.relative_to(root).as_posix()
        sources.append(dict(path=name,sha256=hashlib.sha256(raw).hexdigest()))
        last=0
        for match in re.finditer(r'\\input\{([^}]+)\}',text):
            chunks.append((name,text[last:match.start()],text[:last].count('\n')+1))
            visit(match.group(1),stack+(path,))
            last=match.end()
        chunks.append((name,text[last:],text[:last].count('\n')+1))
    visit('iclr2027_conference.tex')
    tables=[]
    figures=[]
    for name,text,first_line in chunks:
        pattern=r'\\begin\{(table\*?|longtable|figure\*?)\}(.*?)\\end\{\1\}'
        for match in re.finditer(pattern,text,re.S):
            body=match.group(2)
            bucket=figures if match.group(1).startswith('figure') else tables
            data=body.split(r'\midrule',1)[-1]
            item=dict(number=len(bucket)+1,file=name,line=first_line+text[:match.start()].count('\n'),
                      labels=re.findall(r'\\label\{([^}]+)\}',body),
                      caption_tex=braced_argument(body,r'\caption'),
                      tbd_markers=len(re.findall(r'\\expTBD\b',data)),
                      red_numeric_placeholders=len(re.findall(r'\\resultpending\b',data)),
                      red_text_fields=len(re.findall(r'\\res\{',data)))
            if item['tbd_markers'] or item['red_numeric_placeholders'] or item['red_text_fields']:
                item['placeholder_rows_tex']=[r.strip() for r in data.split(r'\\')
                                               if re.search(r'\\(?:expTBD\b|resultpending\b|res\{)',r)]
            bucket.append(item)
    return dict(schema='warm.paper-gap-audit.v1',entrypoint='iclr2027_conference.tex',
                active_sources=sources,tables=tables,figures=figures,
                tbd_table_count=sum(t['tbd_markers']>0 for t in tables),
                tbd_field_markers=sum(t['tbd_markers'] for t in tables),
                red_numeric_table_cells=sum(t['red_numeric_placeholders'] for t in tables),
                red_numeric_figure_cells=sum(t['red_numeric_placeholders'] for t in figures),
                red_text_table_fields=sum(t['red_text_fields'] for t in tables),
                scope='Marker counts, not scalar measurement counts; no experimental values inferred.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--pdf',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.root)
    result['pdf_sha256']=hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'tables','figures','active_sources'}},ensure_ascii=False))
    for table in result['tables']:
        if table['tbd_markers'] or table['red_numeric_placeholders'] or table['red_text_fields']:
            print(json.dumps({k:v for k,v in table.items() if k not in {'caption_tex','placeholder_rows_tex'}},ensure_ascii=False))

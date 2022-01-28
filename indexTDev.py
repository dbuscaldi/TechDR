import json, bz2
import os, sys, os.path
from bs4 import BeautifulSoup

from whoosh import index
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED
from whoosh.analysis import StemmingAnalyzer

schema = Schema(doc_id=ID(stored=True),
                body=TEXT(stored=True, analyzer=StemmingAnalyzer()))

if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

ix = index.create_in("indexdir", schema)

file_path = sys.argv[1]
inf = bz2.BZ2File(file_path, 'r') if file_path.endswith('.bz2') else  open(file_path, 'r')

writer = ix.writer()
i=0
docs = json.load(inf)
for k in docs.keys():
    #print(docs[k])
    data=docs[k]["content"]
    soup = BeautifulSoup(data, 'html.parser')
    text=soup.get_text()
    clean_text=os.linesep.join([s for s in text.splitlines() if s])
    writer.add_document(doc_id=k, body=clean_text)
    if i % 10 == 0:
        sys.stderr.write('.')
    if i % 800 == 0 and i > 0:
        sys.stderr.write('\n')
    sys.stderr.flush()
    i+=1
writer.commit()

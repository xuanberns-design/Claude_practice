import re,sys,glob
bad=0
for f in sorted(glob.glob('0*.tex')):
    s=open(f,encoding='utf8').read()
    # fix: control word immediately followed by CJK / fullwidth punctuation
    s2=re.sub(r'(\\[A-Za-z]+)(?=[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”‘’·])', r'\1{}', s)
    if s2!=s:
        print(f,'fixed',len(re.findall(r'(\\[A-Za-z]+)(?=[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”‘’·])', s)))
        open(f,'w',encoding='utf8').write(s2)
    # \cpp inside tabularx/longtable
    for m in re.finditer(r'\\begin\{(tabularx|longtable)\}.*?\\end\{\1\}', s2, re.S):
        if '\\cpp{' in m.group(0):
            print(f,'WARN \\cpp in',m.group(1),'at',s2[:m.start()].count('\n')+1); bad=1
# circled digits must be wrapped in \ci{}
import re as _re, glob as _g
for f in sorted(_g.glob('0*.tex')):
    s=open(f,encoding='utf8').read()
    s2=_re.sub(r'(?<!\\ci\{)([①-⑩])',r'\\ci{\1}',s)
    if s2!=s: open(f,'w',encoding='utf8').write(s2); print(f,'wrapped circled digits')

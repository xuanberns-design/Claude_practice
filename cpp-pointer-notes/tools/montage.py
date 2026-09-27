import sys,glob
from PIL import Image
# usage: montage.py out.png page1.png page2.png ...
out=sys.argv[1]; ims=[Image.open(p) for p in sys.argv[2:]]
w=sum(i.width for i in ims)+10*(len(ims)-1); h=max(i.height for i in ims)
m=Image.new('RGB',(w,h),'gray'); x=0
for i in ims: m.paste(i,(x,0)); x+=i.width+10
m.save(out)

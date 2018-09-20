import hgtk
import pickle
import sys
import ph_model
import numpy as np
import os

os.environ['LANG'] = 'ko_KR.utf8'
#this codes automatically executed?-yes
wdemb_size=None
hchset=()

#dict load
mwlen =0
hchdct=None
first_exec_flag=1
wdict={}

def vector_preprocessing(vec_file):
    global wdict
    global wdemb_size

    with open(vec_file,'r') as f:
        for i,line in enumerate(f):
            if i==0:
                voclen_embsize=line
                wdemb_size=int(line.split(' ')[1])
                continue
            line = line.replace('\n','')
            line = line.split(' ')
            
            wdkey=i-1
            wd=None
            vec=[]
            for idx,e in enumerate(line):
                if idx==0:
                    wd=e.encode('utf-8')
                    continue
                if e=='':
                    continue
                vec.append(float(e))

            npvec = np.asarray(vec).reshape((1,wdemb_size))
            wdict[wdkey]=[wd,npvec]
    
    #remove latin space character.
    for i in wdict:
        wdict[i][0]=wdict[i][0].decode('utf-8').replace(u'\xa0', u'').encode('utf-8')    
    
#char_vocab making
def hanchar():
    global mwlen
    
    hchset = set()
    for i in wdict:
        if i==0:
            continue
            
        y = wdict[i][0].decode('utf-8').replace(u'\xa0', u'')
        x = hgtk.text.decompose(y)
        if len(x) > mwlen:
            mwlen=len(x)
        
        for z in x:
            hchset.add(z)
            #print(len(hchset))
            
    #print('char voc size : ',len(hchset))
    return hchset

# '1' : start, '2' : end of char, '3' : end of word, 0 : empty space.
def padder(inword):
    global mwlen
    global hchdct
    
    #processing not executed.
    if mwlen==0:
        hanchar()
        
    inword = hgtk.text.decompose(inword)
    
    x = np.zeros(shape=(mwlen,), dtype=int)
    x[0]=1
    n=0
    for n, c in enumerate(inword, start=1):
        x[n] = hchdct[c]
    # last char dosen't contains end of char 2.    
    n+=1
    x[n]=3
    
    return x

def w2v(inword):
    global wdict
    
    for idx, lis in wdict.items():
        if lis[0] == inword.encode('utf-8'):
            return lis[1]

def unk_padder(inword):
    return padder(inword).reshape([1,mwlen])

#after exclude, you need to processing() again for updated batch inpu t& output.
def exclude_word(exword):
    global wdict    
    didx=-1
    for idx, lis in wdict.items():
        if lis[0] == exword.encode('utf-8'):
            didx=idx
            break
    #there's no word
    if didx==-1:
        print('no such word in dict to exclude')
        return

    x = {didx : wdict[didx]}
    del wdict[didx]
    return x

def update_wdict(udict):
    global wdict
    wdict.update(udict)
        
def processing(btch_size=20):
    global mwlen
    global wdict
    global wdemb_size
    global hchdct
    global first_exec_flag
    global hchset
    batch_size=btch_size
    
    #char_vocab dict with end symbol.
    if mwlen==0:
        hchset= hanchar()
    hchset = sorted(hchset)
    #remove 'ᴥ' at position 0.
    #print('this is popped ', hchset[0])
    #print('this is 1 idex ', hchset[1])
    #print('this is hchset ', hchset)
    if hchset[0] == b'\xa0':
        hchset.pop(0)
    if hchset[0] == 'ᴥ':
        hchset.pop(0)
    #hchset.pop(0)
    
    #print('this is hchset2 ', hchset)
    
    hchdct={}
    for i, c in enumerate(hchset, start=4):
        hchdct[c]=i
    hchdct['ᴥ']=2

    #print('longest word length : ',mwlen)
    #start, end character.
    
    #processing first execution. adjust max word length.
    # hard coded.
    if first_exec_flag==1:
        mwlen+=2
        #delete vector for UNK
        del wdict[0]
        first_exec_flag=0
    
    #reduced length for batch size compact.
    reduced_length = (len(wdict) // (batch_size) ) *batch_size
    #print('total dict length :',len(wdict))
    #print('reduced length for batch : ', reduced_length)

    x_batch=np.ndarray(shape=(reduced_length, mwlen), dtype=int)
    y_batch=np.ndarray(shape=(reduced_length, wdemb_size), dtype=float)

    #kinda crazy indexing, dict also has index. so code used messed up, but aligned.
    for wi, i in enumerate(wdict):
        if wi==reduced_length:
            break
        if i not in wdict:
            continue
            
        voc_word = wdict[i][0].decode('utf-8')
        x= padder(voc_word)

        x_batch[wi]=x
        y_batch[wi]=wdict[i][1]

    #reshape for batch size.
    xs=x_batch.reshape([-1, batch_size, mwlen])
    ys=y_batch.reshape([-1, batch_size, wdemb_size])

    return xs,ys

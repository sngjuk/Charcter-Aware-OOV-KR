from gensim.models import KeyedVectors
from subprocess import call, run, PIPE
import subprocess
import sys
import numpy as np

model=None

def oov(inword):
	global model
#	p1 = subprocess.Popen(["cat","queries.txt"], stdout=subprocess.PIPE)
#	p2 = subprocess.Popen(["./fasttext", "print-word-vectors", "kor_model.bin"], stdin=p1.stdout, stdout=subprocess.PIPE)
#	p1.stdout.close()
#	output = p2.communicate()[0]
#	print(output)

	p = run(['python','ph_train.py', '--inference_mode','--model_save_path='+sys.argv[1],'--input_vec='+sys.argv[2] ], stdout=PIPE, input=inword.encode('utf8'))
#	print(p.returncode)
	res =p.stdout
#	print(res)
	res=res.decode('utf-8')	
	vec=[]
	res = res.replace('\n', '')
	res = res.split(' ')
	for i,e in enumerate(res):
		if i==0:
			continue
		if e=='':
			continue
		vec.append(float(e))

	npvec = np.asarray(vec)
	print(inword)
	print(model.similar_by_vector(npvec))


def _start_shell(local_ns=None):
	# An interactive shell is useful for debugging/development.
	import IPython
	user_ns = {}
	if local_ns:
		user_ns.update(local_ns)
	user_ns.update(globals())
	IPython.start_ipython(argv=[], user_ns=user_ns)

def main():
	global model
	print('# Usage: python 2ph_oov_sim.py saved_model_path model.vec')
	if len(sys.argv) < 3:
		print('-Not enough arguments : Expected 2 model_path, .vec')
		sys.exit()
	print('Loading vector ...')
	model = KeyedVectors.load_word2vec_format(sys.argv[2])
	print('#Test E.g.,\n#In [0]: oov("수박아이스크림")\n#In [1]: oov("녹차휴지")')

	_start_shell(locals())

if __name__=="__main__":
	main()



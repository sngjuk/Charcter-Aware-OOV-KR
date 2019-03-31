<b>README</b> <br>
1. Fasttext 모델이 OOV에 대하여 더 강건하고 정확한 결과를 보입니다. <br>
2. 아래의 결과들은 cherry picked 되었기에 본 토이모델 사용을 추천하지 않습니다. <br>
3. 사용된 나무위키데이터(220mb)는 한국어 학습에 유용할 수 있습니다. [google drive](https://drive.google.com/file/d/1--yfaeNHd_xpoJQxdNmTl16_QnhEm1Ma/view?usp=sharing) <br> 

Character-Aware-Language model을 이용한 한국어 자소단위 Out of Vocabulary Vector 학습/추론<br> <a href="https://github.com/mkroutikov/tf-lstm-char-cnn"> mkroutikov </a> model에서 많은 부분을 가져옴.

<b> --Train Usage : </b>
 
```python ph_train.py --input_vec=input.vec --model_save_path=./save_dir/```

<b> Requirements </b> - export LANG=ko_KR.utf8 환경변수 설정, hgtk (pip install hgtk) 자소분리, tensorflow >= 1.0<br>
  <b>Input</b> - input_vec.vec (gensim format)<br>
  <b>Output</b> - Trained Model  <br><t>       [checkpoint, model.ckpt-xxxx.data-00000-of-00001, model.ckpt-xxxx.index, model.ckpt-xxxx.meta]<br>


<b> --Inference Usage : </b> <br>
```python ph_train.py --inference_mode --model_save_path=./save_dir/ --input_vec=input.vec``` 
<br> Inference 모드는 input.vec 사용하지 않지만 구조상 입력.

  <b> Input </b> - OOV 단어 <br>
  <b> Output </b> - 해당 단어에대한 Vector<br>
  *fastText와 동일 format: stdin, stdout <br>

<b> --OOV sim test Usage: </b><br>
``` python ph_oov_test.py ./saved_model/ ./data/NN-numalp-space-ep200em200min24.vec ```<br>
** Similarity test에서는 oov추론시마다 모델을 GPU에 새로 load하기 때문에 실행이 오래걸리지만 실제사용시에는 cat queries.txt | python --- 과 같이 pipe를 이용해 여러 oov 를 한번에 전달하고 결과를 한번에 받을 수 있음. <br>

<img src="http://pds21.egloos.com/pds/201809/20/00/c0134200_5ba383ed5f18f.png"><br>
<img src="http://pds27.egloos.com/pds/201712/28/00/c0134200_5a447d9ddf353.png">

<b> 결과/분석 </b><br>
1. 어렵지 않은 단어에 대해서는 나쁘지 않은 추론을 하는 것으로 보이나 수박아이스크림(수박/아이/스크림, 수박/아이스/크림)과 같이 중의적인 의미의 단어에 대해 낮은 성능 보임.
2. loss 낮을수록 OOV의 의미를 추론하는 성능저하.
3. 현재는 자소 구성중심으로 벡터를 만듦. Highway network 추가하는것으로 의미에 비중을 둔 추론을 할 수 있을것.
4. 현재까지는 본 모델보다 FastText 추론이 robust한 결과를 보임. 나무위키 임베딩데이터 외에는 실험하지 못했음.
<br>
<br>
<b> 향후과제 </b><br>
1. parameter 실험, 음소 아닌 음절단위로 학습.
2. 480MB의 나무위키 데이터가 아닌 다른 데이터로 학습.
3. 임시 검증방법 수정.
<br>

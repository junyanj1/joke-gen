# joke-gen

## Origin
Generating jokes with SeqGAN. See the SeqGAN paper: [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473.pdf).

## Project Structure
#### /SeqGAN
- joke generation with SeqGAN simple implementation (no rollout) 
- original repo: [https://github.com/suragnair/seqGAN](https://github.com/suragnair/seqGAN)
```
$ pip insall -r requirements.txt
$ cd SeqGAN/
$ python main.py
```
#### /SeqGAN-rollout
- joke generation with SeqGAN rollout implementation 
- original repo: [https://github.com/ZiJianZhao/SeqGAN-PyTorch](https://github.com/ZiJianZhao/SeqGAN-PyTorch)
```
$ pip insall -r requirements.txt
$ cd SeqGAN-rollout/
$ python main.py
```
#### /ColBERT humor 
- evaluation framework using ColBERT model and BLEU scores



    


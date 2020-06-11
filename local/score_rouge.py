import sys
import json
import jieba
import numpy as np
from rouge import Rouge

rouge = Rouge()
cjk_ranges = [
        ( 0x4E00,  0x62FF),
        ( 0x6300,  0x77FF),
        ( 0x7800,  0x8CFF),
        ( 0x8D00,  0x9FCC),
        ( 0x3400,  0x4DB5),
        (0x20000, 0x215FF),
        (0x21600, 0x230FF),
        (0x23100, 0x245FF),
        (0x24600, 0x260FF),
        (0x26100, 0x275FF),
        (0x27600, 0x290FF),
        (0x29100, 0x2A6DF),
        (0x2A700, 0x2B734),
        (0x2B740, 0x2B81D),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x2F800, 0x2FA1F),
        (ord('0'), ord('9')),
        (ord('a'), ord('z')),
        (ord('A'), ord('Z'))
    ]

def preproc(string, hyp=False):
    def is_important(char):
        if char == ' ':
            return True
        char = ord(char)
        for bottom, top in cjk_ranges:
            if char >= bottom and char <= top:
                return True
        return False
    
    if hyp:
        string = string[:-5]
    return ''.join(c for c in string if is_important(c))

def jieba_split(string, hyp=False):
    return [w for w in jieba.cut(preproc(string, hyp)) if w != ' ']

def jieba_rouge(hyp_ref_tuple):
    try:
        hypothesis, reference = hyp_ref_tuple
        hypothesis = ' '.join(jieba_split(hypothesis, hyp=True))
        reference = ' '.join(jieba_split(reference))
        scores = rouge.get_scores(hypothesis, reference)[0]
        rouge_1f, rouge_2f, rouge_lf = scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']
        rouge_1r, rouge_2r, rouge_lr = scores['rouge-1']['r'], scores['rouge-2']['r'], scores['rouge-l']['r']
        rouge_1p, rouge_2p, rouge_lp = scores['rouge-1']['p'], scores['rouge-2']['p'], scores['rouge-l']['p']
        return (rouge_1f, rouge_2f, rouge_lf, rouge_1r, rouge_2r, rouge_lr, rouge_1p, rouge_2p, rouge_lp)
    except:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)

def char_rouge(hyp_ref_tuple):
    try:
        hypothesis, reference = hyp_ref_tuple
        hypothesis = ' '.join(c for c in preproc(hypothesis, hyp=True))
        reference = ' '.join(c for c in preproc(reference))
        scores = rouge.get_scores(hypothesis, reference)[0]
        rouge_1f, rouge_2f, rouge_lf = scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']
        rouge_1r, rouge_2r, rouge_lr = scores['rouge-1']['r'], scores['rouge-2']['r'], scores['rouge-l']['r']
        rouge_1p, rouge_2p, rouge_lp = scores['rouge-1']['p'], scores['rouge-2']['p'], scores['rouge-l']['p']
        return (rouge_1f, rouge_2f, rouge_lf, rouge_1r, rouge_2r, rouge_lr, rouge_1p, rouge_2p, rouge_lp)
    except:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)

def show_rouge(rouge_list):
    rouge_metrics = zip(*rouge_list)
    rouge_1f, rouge_2f, rouge_lf, rouge_1r, rouge_2r, rouge_lr, rouge_1p, rouge_2p, rouge_lp = (np.mean(metric) for metric in rouge_metrics)
    print('-----------------------')
    print('  ROUGE-1(F): %.4f' % (100*rouge_1f))
    print('  ROUGE-2(F): %.4f' % (100*rouge_2f))
    print('  ROUGE-L(F): %.4f' % (100*rouge_lf))
    print('-----------------------')
    print('  ROUGE-1(R): %.4f' % (100*rouge_1r))
    print('  ROUGE-2(R): %.4f' % (100*rouge_2r))
    print('  ROUGE-L(R): %.4f' % (100*rouge_lr))
    print('-----------------------')
    print('  ROUGE-1(P): %.4f' % (100*rouge_1p))
    print('  ROUGE-2(P): %.4f' % (100*rouge_2p))
    print('  ROUGE-L(P): %.4f' % (100*rouge_lp))
    print('-----------------------')
    
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(sys.argv)
        print('Usage: python3 score_rouge.py <data.json>')
        exit(1)
        
    jieba.set_dictionary('local/dict.txt.big')
    jieba.initialize()
    
    # Load json-format decoding
    print('Loading json decoding file...')
    file_path = sys.argv[1]
    data = json.load(open(file_path, encoding='utf-8'))
    utt_ids = list(data['utts'].keys())
    
    # Character-based
    rouge_list = []
    for i, utt_id in enumerate(utt_ids, start=1):
        hypothesis = data['utts'][utt_id]['output'][0]['rec_text']
        reference = data['utts'][utt_id]['output'][0]['text']
        hyp_ref_tuple = (hypothesis, reference)
        rouge_list.append(char_rouge(hyp_ref_tuple))
        print('Character-based: scoring %d/%d(%.2f%%)\r' % (i, len(utt_ids), 100*i/len(utt_ids)), end='')
    print()
    show_rouge(rouge_list)

    # Word-based ROUGE (segmented by Jieba)
    rouge_list = []
    for i, utt_id in enumerate(utt_ids, start=1):
        hypothesis = data['utts'][utt_id]['output'][0]['rec_text']
        reference = data['utts'][utt_id]['output'][0]['text']
        hyp_ref_tuple = (hypothesis, reference)
        rouge_list.append(jieba_rouge(hyp_ref_tuple))
        print('Word-based: scoring %d/%d(%.2f%%)\r' % (i, len(utt_ids), 100*i/len(utt_ids)), end='')
    print()
    show_rouge(rouge_list)

import os
import re
from sklearn.model_selection import train_test_split

def is_chinese(cp):
    cp = ord(cp)
    if ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F)
            or (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF)
            or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)):
        return True
    return False

def all_chinese(string):
    return string and all(is_chinese(chn) for chn in string)

def remove_punct_blank(value):
    result = re.sub('\W+', '', value).replace("_", '')
    return result


MINLEN = 9

if __name__ == '__main__':
    print('`preprocess_data.py` starts running.')
    
    data_dir = './data/all'
    output_dir = './data/all_split'
    os.makedirs(output_dir)
    subdata_ids = sorted(os.listdir(data_dir))
    
    for n, subdata_id in enumerate(subdata_ids, start=1):
        print('Preprocessing subdata %s... (%d/%d)' % (subdata_id, n, len(subdata_ids)))
        subdata_dir = os.path.join(data_dir, subdata_id)

        # Init utterance relations
        utt_list = []
        wav_set_train, wav_set_dev, wav_set_test = set(), set(), set()
        utt2text, utt2wav, utt2segment, wav2path = {}, {}, {}, {}

        # Parse `text`
        text_fpath = os.path.join(subdata_dir, 'text')
        lines = open(text_fpath, encoding='utf-8').readlines()
        for i, line in enumerate(lines, start=1):
            print(' > Parsing `text`: %d/%d(%.2f%%)\r' % (i, len(lines), 100*i/len(lines)), end='')
            utt_id, text = line.strip().split(' ', 1)
            text = remove_punct_blank(text)
            if len(text) < MINLEN or not all_chinese(text):
                continue
            utt_list.append(utt_id)
            utt2text[utt_id] = text
        print()

        # Split utterances to 0.98/0.01/0.01 as train/dev/test set
        utt_list_train, utt_list_eval = train_test_split(utt_list, test_size=0.02, random_state=42)
        utt_list_dev, utt_list_test = train_test_split(utt_list_eval, test_size=0.5, random_state=42)
        utt_set_train = set(utt_list_train)
        utt_set_dev = set(utt_list_dev)
        utt_set_test = set(utt_list_test)

        # Parse `segments`
        segments_fpath = os.path.join(subdata_dir, 'segments')
        lines = open(segments_fpath, encoding='utf-8').readlines()
        for i, line in enumerate(lines, start=1):
            print(' > Parsing `segments`: %d/%d(%.2f%%)\r' % (i, len(lines), 100*i/len(lines)), end='')
            utt_id, wav_id, segment = line.strip().split(' ', 2)
            if utt_id in utt_set_dev:
                wav_set_dev.add(wav_id)
            elif utt_id in utt_set_test:
                wav_set_test.add(wav_id)
            elif utt_id in utt_set_train:
                wav_set_train.add(wav_id)
            else:
                continue
            utt2wav[utt_id] = wav_id
            utt2segment[utt_id] = segment
        print()

        # Parse `wav.scp`
        wavscp_fpath = os.path.join(subdata_dir, 'wav.scp')
        lines = open(wavscp_fpath, encoding='utf-8').readlines()
        for i, line in enumerate(lines, start=1):
            print(' > Parsing `wav.scp`: %d/%d(%.2f%%)\r' % (i, len(lines), 100*i/len(lines)), end='')
            wav_id, wav_path = line.strip().split(' ', 1)
            wav2path[wav_id] = wav_path
        print()

        # Save files to split
        for split, utt_list_split,  wav_set_split in \
                zip(['dev', 'test', 'train'], \
                [utt_list_dev, utt_list_test, utt_list_train], \
                [wav_set_dev, wav_set_test, wav_set_train]):
            print('> Saving files of %s split' % split)
            save_dir = os.path.join(output_dir, '%s_%s' % (subdata_id, split))
            os.makedirs(save_dir)
            
            f_text     = open(os.path.join(save_dir, 'text')    , mode='w', encoding='utf-8')
            f_segments = open(os.path.join(save_dir, 'segments'), mode='w', encoding='utf-8')
            f_wavscp   = open(os.path.join(save_dir, 'wav.scp') , mode='w', encoding='utf-8')
            f_utt2spk  = open(os.path.join(save_dir, 'utt2spk') , mode='w', encoding='utf-8')
            
            for utt_id in utt_list_split:
                text, wav_id, segment = utt2text[utt_id], utt2wav[utt_id], utt2segment[utt_id]
                f_text.write('%s %s\n' % (utt_id, text))
                f_segments.write('%s %s %s\n' % (utt_id, wav_id, segment))
                f_utt2spk.write('%s %s\n' % (utt_id, utt_id))
                
            for wav_id in wav_set_split:
                wav_path = wav2path[wav_id]
                f_wavscp.write('%s %s\n' % (wav_id, wav_path))
                
    print('`preprocess_data.py` finished successfully.')

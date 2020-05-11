import os


if __name__ == '__main__':
    data_dir = '/home/M10815022/Dataset/Taiwanese_ASR'
    subdata_ids = sorted(os.listdir(data_dir))  # N=12
    
    # Training set
    #for dset, sub_idx in zip(('train', 'dev', 'test'), ((8), 7, 5)):
    for subdata_id in subdata_ids:
        
        print('Preparing subdata %s ...' % subdata_id)
        dset_dir = 'data/all/%s/' % subdata_id
        os.makedirs(dset_dir)
        
        f_text     = open('%s/text'     % dset_dir, mode='w')
        f_segments = open('%s/segments' % dset_dir, mode='w')
        f_wavscp   = open('%s/wav.scp'  % dset_dir, mode='w')
        f_utt2spk  = open('%s/utt2spk'  % dset_dir, mode='w')
    
        #subdata_id = subdata_ids[sub_idx]
        subdata_dir = os.path.join(data_dir, subdata_id)
        split_ids = sorted(os.listdir(subdata_dir))  # M=30+
               
        for i, split_id in enumerate(split_ids, start=1): 
            split_dir = os.path.join(subdata_dir, split_id)
            os.listdir(split_dir)

            # text
            text_fpath = os.path.join(split_dir, 'text')
            with open(text_fpath) as f:
                text = f.read()
            f_text.write(text)

            # segments
            segments_fpath = os.path.join(split_dir, 'segments')
            with open(segments_fpath) as f:
                segments = f.read()
            f_segments.write(segments)
            
            # wav.scp
            audio_fname = None
            for fname in os.listdir(split_dir):
                if fname.endswith('.wav'):
                    audio_fname = fname
            assert audio_fname is not None
            audio_fpath = os.path.join(split_dir, audio_fname)
            f_wavscp.write('%s %s\n' % (split_id, audio_fpath))

            # utt2spk
            lines = segments.rstrip().split('\n')
            for line in lines:
                utt_id, _, _, _ = line.split()
                f_utt2spk.write('%s %s\n' % (utt_id, utt_id))

            print('Processed split: %d/%d\r' % (i, len(split_ids)), end='')
        print()
    print('`prepare_data.py` finished successfully.')

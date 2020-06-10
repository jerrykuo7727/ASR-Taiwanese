import os


if __name__ == '__main__':
    print('`prepare_data.py` starts running.')
    
    data_dir = '/home/M10815022/Dataset/Taiwanese_ASR'
    subdata_ids = sorted(os.listdir(data_dir))
    
    # Training set
    for n, subdata_id in enumerate(subdata_ids, start=1):
        print('Preparing subdata %s... (%d/%d)' % (subdata_id, n, len(subdata_ids)))
        dset_dir = 'data/all/%s/' % subdata_id
        os.makedirs(dset_dir)
        
        f_text     = open('%s/text'     % dset_dir, mode='w')
        f_segments = open('%s/segments' % dset_dir, mode='w')
        f_wavscp   = open('%s/wav.scp'  % dset_dir, mode='w')
        f_utt2spk  = open('%s/utt2spk'  % dset_dir, mode='w')
    
        subdata_dir = os.path.join(data_dir, subdata_id)
        split_ids = sorted(os.listdir(subdata_dir))
               
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

            print(' > Processed split: %d/%d(%.2f%%)\r' % (i, len(split_ids), 100*i/len(split_ids)), end='')
        print()
    print('`prepare_data.py` finished successfully.')

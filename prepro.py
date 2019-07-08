# -*- coding: utf-8 -*-

import os
import errno
import sentencepiece as spm
import re
from hparams import Hparams
import logging

logging.basicConfig(level=logging.INFO)

def prepro(hp):
    """Load raw data -> Preprocessing -> Segmenting with sentencepice
    hp: hyperparams. argparse.
    """
    logging.info("# Check if raw files exist")
    train1 = "TED_data/train.en"
    train2 = "TED_data/train.zh"
    eval1 = "TED_data/valid.en"
    eval2 = "TED_data/valid.zh"
    test1 = "TED_data/test.en"
    test2 = "TED_data/test.zh"
    for f in (train1, train2, eval1, eval2, test1, test2):
        if not os.path.isfile(f):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)

    logging.info("# Preprocessing")
    # train
    _prepro = lambda x:  [line.strip() for line in open(x, 'r').read().split("\n")]
    prepro_train1, prepro_train2 = _prepro(train1), _prepro(train2)
    assert len(prepro_train1)==len(prepro_train2), "Check if train source and target files match."

    # eval
    _prepro = lambda x: [line.strip() for line in open(x, 'r').read().split("\n")]
    prepro_eval1, prepro_eval2 = _prepro(eval1), _prepro(eval2)
    assert len(prepro_eval1) == len(prepro_eval2), "Check if eval source and target files match."

    # test
    prepro_test1, prepro_test2 = _prepro(test1), _prepro(test2)
    assert len(prepro_test1) == len(prepro_test2), "Check if test source and target files match."

    logging.info("Let's see how preprocessed data look like")
    logging.info("prepro_train1:", prepro_train1[0])
    logging.info("prepro_train2:", prepro_train2[0])
    logging.info("prepro_eval1:", prepro_eval1[0])
    logging.info("prepro_eval2:", prepro_eval2[0])
    logging.info("prepro_test1:", prepro_test1[0])
    logging.info("prepro_test2:", prepro_test2[0])

    logging.info("# write preprocessed files to disk")
    os.makedirs("TED_data/prepro", exist_ok=True)
    def _write(sents, fname):
        with open(fname, 'w') as fout:
            fout.write("\n".join(sents))

    _write(prepro_train1, "TED_data/prepro/train.en")
    _write(prepro_train2, "TED_data/prepro/train.zh")
    _write(prepro_train1+prepro_train2, "TED_data/prepro/train")
    _write(prepro_eval1, "TED_data/prepro/eval.en")
    _write(prepro_eval2, "TED_data/prepro/eval.zh")
    _write(prepro_test1, "TED_data/prepro/test.en")
    _write(prepro_test2, "TED_data/prepro/test.zh")

    logging.info("# Train a joint BPE model with sentencepiece")
    os.makedirs("TED_data/segmented", exist_ok=True)
    train = '--input=TED_data/prepro/train --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=TED_data/segmented/bpe --vocab_size={} \
             --model_type=bpe'.format(hp.vocab_size)
    spm.SentencePieceTrainer.Train(train)

    logging.info("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()
    sp.Load("TED_data/segmented/bpe.model")

    logging.info("# Segment")
    def _segment_and_write(sents, fname):
        with open(fname, "w") as fout:
            for sent in sents:
                pieces = sp.EncodeAsPieces(sent)
                fout.write(" ".join(pieces) + "\n")

    _segment_and_write(prepro_train1, "TED_data/segmented/train.en.bpe")
    _segment_and_write(prepro_train2, "TED_data/segmented/train.zh.bpe")
    _segment_and_write(prepro_eval1, "TED_data/segmented/eval.en.bpe")
    _segment_and_write(prepro_eval2, "TED_data/segmented/eval.zh.bpe")
    _segment_and_write(prepro_test1, "TED_data/segmented/test.en.bpe")

    logging.info("Let's see how segmented data look like")
    print("train1:", open("TED_data/segmented/train.en.bpe",'r').readline())
    print("train2:", open("TED_data/segmented/train.zh.bpe", 'r').readline())
    print("eval1:", open("TED_data/segmented/eval.en.bpe", 'r').readline())
    print("eval2:", open("TED_data/segmented/eval.zh.bpe", 'r').readline())
    print("test1:", open("TED_data/segmented/test.en.bpe", 'r').readline())

if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
    logging.info("Done")

import os, time, re, sys, math
from collections import defaultdict, OrderedDict, Counter
import core.utils.common as common


class VocabularyBase(object):
  def __init__(self, pad_token='<pad>', unk_token='<unk>'):
    self.vocab = None
    self.rev_vocab = None
    self.start_vocab = [pad_token, unk_token]
    self.PAD = pad_token
    self.UNK = unk_token

  @property
  def size(self):
    return len(self.vocab)

  @property
  def PAD_ID(self):
    return self.token2id(self.PAD)

  @property
  def UNK_ID(self):
    return self.token2id(self.UNK)

  def id2token(self, _id):
    if type(_id) not in [int, np.int32, np.int64]:
      sys.stderr.write(str(type(_id)))
      raise ValueError('Token ID must be an integer.')
    elif _id < 0 or _id > len(self.rev_vocab):
      return self.UNK
    elif _id == self.PAD_ID:
      return ''
    else:
      return self.rev_vocab[_id]

  def token2id(self, token):
    return self.vocab.get(token, self.vocab.get(self.UNK, None))


class LabelVocabulary(VocabularyBase):
  '''
  A class to manage transformation between feature tokens and their ids.
  '''
  def __init__(self, all_tokens, pad_token='<pad>', unk_token='<unk>'):
    super(LabelVocabulary, self).__init__(pad_token=pad_token, 
                                            unk_token=unk_token)

    counter = Counter(all_tokens)
    self.freq = counter.values
    self.rev_vocab = self.start_vocab + list(counter.keys())
    self.vocab = OrderedDict([(t, i) for i,t in enumerate(self.rev_vocab)])

  def __str__(self):
    return '<%s>: ' % self.__class__ + str(self.rev_vocab[:5] + ['...']) 

  def tokens2ids(self, tokens):
    assert type(tokens) in [list, tuple] and type(tokens[0]) == str
    return [self.token2id(t) for t in tokens]

  def ids2tokens(self, ids):
    return [self.id2token(_id) for _id in ids]

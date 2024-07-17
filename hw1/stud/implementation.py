import numpy as np
from typing import List
import tqdm
import torch 
from torch import nn
from torch.utils.data import Dataset


from model import Model


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    #return RandomBaseline()
	return StudentModel(device)

class RandomBaseline(Model):
    options = [
        (22458, "B-ACTION"),
        (13256, "B-CHANGE"),
        (2711, "B-POSSESSION"),
        (6405, "B-SCENARIO"),
        (3024, "B-SENTIMENT"),
        (457, "I-ACTION"),
        (583, "I-CHANGE"),
        (30, "I-POSSESSION"),
        (505, "I-SCENARIO"),
        (24, "I-SENTIMENT"),
        (463402, "O")
    ]
	
    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model):
	
	# STUDENT: construct here your model
	# this class should be loading your weights and vocabulary
	
	
	def __init__(self,device):
		#super(StudentModel,self).__init__()
		self.device = device
		#self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.vocabs = self.get_vocabs('model/vocabs.txt')
		self.tags = self.get_tags('model/tags.txt')
		self.START_TAG = "<START>"
		self.STOP_TAG = "<STOP>"


		self.tags_tokenizer = self.tokenizer(self.tags,tagging=True)
		self.index_tag = self.indexes(self.tags_tokenizer)
		self.tag_to_ix = self.tags_tokenizer

		self.tag_to_ix["<START>"]=len(self.tag_to_ix)
		self.tag_to_ix["<STOP>"]=len(self.tag_to_ix)
    
	
		self.vocabs_tokenizer = self.tokenizer(self.vocabs)
		self.index_vocab = self.indexes(self.vocabs_tokenizer)
		
		
		
		self.VOCAB_SIZE = len(self.vocabs) + 2 # 2 is (UNK, PADDING) values
		self.vocab_size = self.VOCAB_SIZE
		self.embedding_dim = 100
		self.hidden_dim = 128

		self.model = self.Model_CRF(self.vocab_size, self.tag_to_ix, self.embedding_dim, self.hidden_dim,self.device,self.START_TAG,self.STOP_TAG).to(self.device)
		if device == 'cpu':
			self.model.load_state_dict(torch.load('model/model.pth'),strict=True)
		else:
			self.model.load_state_dict(torch.load('model/model.pth'))
		self.model.eval()
		
    
	def tokens_lower(self,tokens_values):
		tokens = []
		for num in tqdm(range(len(tokens_values))):
			tokens.append(list(map(str.lower,tokens_values[num])))
		return tokens
	
	def get_vocabs(self,file_path_vocab):
		vocabs = []
		
		with open(file_path_vocab, 'r') as fp:
			for line in fp:
				x = line[:-1]
				vocabs.append(x)
		return vocabs


	def get_tags(self,file_path_tags):
		tags = []
		
		with open(file_path_tags, 'r') as fp:
			for line in fp:         
				x = line[:-1]        
				tags.append(x)
		return tags
  	
    
	def tokenizer(self,unique_values,padding_token='__PADDING__',OOV_token='__UNK__',tagging=False):
		tokenized_val = {}
		if tagging == True:
			pass
		else:
			tokenized_val[OOV_token]=1
		tokenized_val[padding_token]=0
		value = len(tokenized_val)
		for i in unique_values:
			tokenized_val[i]=value
			value+=1
		
		return tokenized_val
  	
	def indexes(self,tokens):
		idx = {j:i for i,j in tokens.items()}
		return idx
    	
	def data2idx_vocabs(self,lst_vocabs, tokens):
		data2idx = []
		for vocabs in lst_vocabs:
			data = []
			for vocab in vocabs:
				if vocab in tokens:
					data.append(tokens[vocab])
				else:
					data.append(tokens['__UNK__'])
			data2idx.append(data)
		return data2idx
  	
	def data2idx_tags(self,lst_tags, tokens):
		data2idx = []
		for tags in lst_tags:
			data = []
			for tag in tags:
				data.append(tokens[tag])
				data2idx.append(data)
		return data2idx
  	
	def padding_sequence(self,list_data,maxlen):
		padded_list = []
		for data in list_data:
			pad_value = maxlen - len(data)
			pad = [0]*pad_value+data
			padded_list.append(np.array(pad,dtype='int32'))
		return np.array(padded_list,dtype='int32')
  	
	

	class Build_Dataset(Dataset):
		def __init__(self, data):
			self.data = data
			
		def __len__(self):
			return len(self.data)
		
		def __getitem__(self, idx):
			sentence = self.data[idx]
			return torch.tensor(sentence)

	def predict(self, tokens: List[List[str]]) -> List[List[str]]:
			# STUDENT: implement here your predict function
			# remember to respect the same order of tokens!
		self.sequences = self.data2idx_vocabs(tokens,self.vocabs_tokenizer)

		self.val = self.padding_sequence(self.sequences,343)
		self.sentence = self.Build_Dataset(self.val)


		with torch.no_grad():
			pred_test = []
			#true_test = []
			for batch_idx_v, sentences_v in enumerate(self.sentence):
				#print(sentences_v)
				sentences_v = sentences_v.to(self.device)
				outputs_v = self.model(sentences_v)
					#true_test.append(tags_v)
				pred_test.append(outputs_v)

	
			pred_tags = []
			for i in range(len(pred_test)):
		
				test_p = pred_test[i][1]
				tags = [self.index_tag[int(j)] for j in list(test_p)][-	len(self.sequences[i]):]
				pred_tags.append(tags)
				
		return pred_tags
		
	class Model_CRF(nn.Module):
		def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,device,start,stop):

			super().__init__()
			self.START_TAG = start
			self.STOP_TAG = stop
			self.device = device
			self.embedding_dim = embedding_dim
			self.hidden_dim = hidden_dim
			self.vocab_size = vocab_size
			self.tag_to_ix = tag_to_ix
			self.tagset_size = len(tag_to_ix)

			self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
			self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
		                    num_layers=1, bidirectional=True)

		
			self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

		
			self.transitions = nn.Parameter(
		    torch.randn(self.tagset_size, self.tagset_size)).to(self.device)

		
			self.transitions.data[tag_to_ix[self.START_TAG], :] = -10000
			self.transitions.data[:, tag_to_ix[self.STOP_TAG]] = -10000

			self.hidden = self.init_hidden()

		def argmax(self,vector):
			_, index = torch.max(vector, 1)
			return index.item()
	
		def log_sum_exp(self,vector):
			maximum_score = vector[0, self.argmax(vector)]
			maximum_score_broadcast = maximum_score.view(1, -1).expand(1, vector.size()[1])
			return maximum_score + torch.log(torch.sum(torch.exp(vector - maximum_score_broadcast).to(self.device)).to(self.device))

		def init_hidden(self):
			return (torch.randn(2, 1, self.hidden_dim // 2).to(self.device),
		        torch.randn(2, 1, self.hidden_dim // 2).to(self.device))

		def _forward_alg(self, feats):
			
			init_alphas = torch.full((1, self.tagset_size), -10000.).to(self.device)
		
			init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

		
			forward_var = init_alphas

		
			for feat in feats:
				alphas_t = []  
				for next_tag in range(self.tagset_size):
					emit_score = feat[next_tag].view(
		            1, -1).expand(1, self.tagset_size)
					trans_score = self.transitions[next_tag].view(1, -1)
					next_tag_var = forward_var + trans_score + emit_score
					alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
					
				forward_var = torch.cat(alphas_t).view(1, -1).to(self.device)
			terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
			alpha = self.log_sum_exp(terminal_var)
			return alpha
		
		def _get_lstm_features(self, sentence):
			self.hidden = self.init_hidden()
			embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
			lstm_out, self.hidden = self.lstm(embeds, self.hidden)
			lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
			lstm_feats = self.hidden2tag(lstm_out)
			return lstm_feats
		
		def _score_sentence(self, feats, tags):
		
			score = torch.zeros(1).to(self.device)
			tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long).to(self.device), tags])
			for i, feat in enumerate(feats):
				score = score + \
					self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
			score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
			return score
		
		def _viterbi_decode(self, feats):
			backpointers = []

		
			init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
			init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

		
			forward_var = init_vvars
			for feat in feats:
				bptrs_t = []
				viterbivars_t = []  
				
				for next_tag in range(self.tagset_size):
					next_tag_var = forward_var + self.transitions[next_tag]
					best_tag_id = self.argmax(next_tag_var)
					bptrs_t.append(best_tag_id)
					viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
					
				forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1).to(self.device)
				backpointers.append(bptrs_t)
				
			terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
			best_tag_id = self.argmax(terminal_var)
			path_score = terminal_var[0][best_tag_id]

		
			best_path = [best_tag_id]
			for bptrs_t in reversed(backpointers):
				best_tag_id = bptrs_t[best_tag_id]
				best_path.append(best_tag_id)
				
			start = best_path.pop()
			assert start == self.tag_to_ix[self.START_TAG]
			best_path.reverse()
			return path_score, best_path
		
		def neg_log_likelihood(self, sentence, tags):
			feats = self._get_lstm_features(sentence)
			forward_score = self._forward_alg(feats)
			gold_score = self._score_sentence(feats, tags)
			return forward_score - gold_score
		
		def forward(self, sentence):  
			lstm_feats = self._get_lstm_features(sentence)

		
			score, tag_seq = self._viterbi_decode(lstm_feats)
			return score, tag_seq
		
		
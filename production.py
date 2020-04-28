# -*- coding: utf-8 -*-
import sys, getopt, json, io
import torch
from transformers import BertTokenizer, BertForQuestionAnswering



def main(argv):
	inputfile = ''#test_data
	model_path = ''#model

	try:
		opts, args = getopt.getopt(argv,"h:m:")
	except getopt.GetoptError:
		print('production.py -m <Model>')
		sys.exit(2)
	print(opts)
	for opt, arg in opts:

		if opt == '-h':
			print('production.py')
			sys.exit()
		elif opt == '-m':
			model_path = arg

	print('載入已訓練的模型： ', model_path)

	with io.open('FGC final/FGC_release_A.json' , 'r', encoding = 'utf-8') as reader:
		q_jf = json.loads(reader.read())

	with io.open('FGC final/FGC_release_A_answers.json' , 'r', encoding = 'utf-8') as reader:
		a_jf = json.loads(reader.read())





	tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
	model = torch.load(model_path, map_location=torch.device('cpu'))#.pkl
	# model = BertForQuestionAnswering.from_pretrained(model_path)#.bin
# "cuda:0" if torch.cuda.is_available() else 
	device = torch.device("cpu")
	print("device:", device)
	model = model.to(device)

	model.eval()
	ans_lst = []
	for a in a_jf:

		for aa in a['QUESTIONS']:

			ans_lst.append(aa["ANSWER"])

		
	#topic_cnt = 0 #debug用early stop參數
	acc = 0
	cnt = 0
	ix = 0
	for d in q_jf:

		print('Passage :\t', d['DTEXT'])

		for q in d['QUESTIONS']:
			#print(a['answer_start'])
			#print('Answer :\t', a['text'])
			question = q['QTEXT'][2:]
			text = d['DTEXT']
			text_corrupt ="[CLS] " + question + " [SEP] " + text
			input_text = text_corrupt[:500] +" [SEP]"  #model limit 512
			#Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]: 0 corresponds to a sentence A token, 1 corresponds to a sentence			
			input_ids = tokenizer.encode(input_text)
			token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]

			#pre_len = len(tokenizer.encode("[CLS] " + question + " [SEP] "))
			#start_positions = pre_len + a['answer_start']
			#end_positions = pre_len + a['answer_start'] + len(tokenizer.encode(a['text']))

				#print('Emb_id :\t',[input_ids])
				#print('Seg_tok :\t',[token_type_ids])
				#print('P_start :\t',start_positions)
				#print('P_end :\t',end_positions)
				#print('A_span :\t'+ ''.join(tokenizer.convert_ids_to_tokens(input_ids[start_positions:end_positions])))
				#print('='*50)
	
				

			cnt+=1
			start_scores, end_scores = model(torch.tensor([input_ids]).to(device), token_type_ids=torch.tensor([token_type_ids]).to(device))
			all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
			ans_pred = ''.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)])
			
			if ans_pred == '':

				text_corrupt ="[CLS] " + question + " [SEP] " + text[-480:]
				input_text = text_corrupt +" [SEP]"  #model limit 512
				#Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]: 0 corresponds to a sentence A token, 1 corresponds to a sentence			
				input_ids = tokenizer.encode(input_text)
				token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]

				
				start_scores, end_scores = model(torch.tensor([input_ids]).to(device), token_type_ids=torch.tensor([token_type_ids]).to(device))
				all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
				ans_pred = ''.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)])




			print('Answer :\t', ans_lst[ix])
			print('Prediction :\t', ans_pred)
			ans_pred = ans_pred.replace('##', '').replace('[UNK]', '').replace('[SEP]', '').strip("。（），、──；？《》〈〉！──……：「」『』 	")
				   
			if  ans_pred.strip("。（），、──；？《》〈〉！──……：「」『』 	#") in ans_lst[ix] and ans_pred != '':
				acc+=1
				print('='*100 + 'ok!')  
			else:
				print('='*100)  
			ix+=1

	if cnt!= 0:
		print("Acc =\t", acc/cnt)

if __name__ == "__main__":
	main(sys.argv[1:])
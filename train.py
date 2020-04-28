# -*- coding: utf-8 -*-
import sys, getopt, json, io
import torch
from transformers import BertTokenizer, BertForQuestionAnswering



def main(argv):
	inputfile = ''#dataset
	outputfile = ''#model
	epoch = 0
	learning_rate = 0
	try:
		opts, args = getopt.getopt(argv,"hi:o:e:l:")
	except getopt.GetoptError:
		print('train.py -i <InputFile> -o <OutputFile> -e <Epoch> -l <LearningRate>')
		sys.exit(2)
	#print(opts)
	for opt, arg in opts:

		if opt == '-h':
			print('train.py -i <InputFile> -o <OutputFile> -e <Epoch> -l <LearningRate>')
			sys.exit()
		elif opt == '-i':
			inputfile = arg
		elif opt == '-o':
			outputfile = arg
		elif opt == '-e':
			epoch = int(arg)
		elif opt == '-l':
			learning_rate = float(arg)

	print('輸入的訓練資料： ', inputfile)
	print('輸出的模型 ： ', outputfile)
	print('Epoch : ', epoch)
	print('LearningRate : ', learning_rate)

	with io.open(inputfile , 'r', encoding = 'utf-8') as reader:
		jf = json.loads(reader.read())

	with io.open('FGC_release_B.json' , 'r', encoding = 'utf-8') as reader:
		q_jfB = json.loads(reader.read())

	with io.open('FGC_release_B_answers.json' , 'r', encoding = 'utf-8') as reader:
		a_jfB = json.loads(reader.read())

	with io.open('FGC_release_A.json' , 'r', encoding = 'utf-8') as reader:
		q_jfA = json.loads(reader.read())

	with io.open('FGC_release_A_answers.json' , 'r', encoding = 'utf-8') as reader:
		a_jfA = json.loads(reader.read())

	# with io.open('customSet.json' , 'r', encoding = 'utf-8') as reader:
	# 	our_jf = json.loads(reader.read())
	
	ans_lstB = []
	ans_lstA = []

	for a in a_jfB:
		ans_lstB.append(a["ANSWER"])
	for a in a_jfA:
		ans_lstA.append(a["ANSWER"])


	tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
	#model = BertForQuestionAnswering.from_pretrained('bert-base-chinese')
	# model = torch.load("AI1.pkl")#.pkl
	model = BertForQuestionAnswering.from_pretrained('hfl/chinese-roberta-wwm-ext-large')#.bin

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("device:", device)
	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	model.train()

	def find_sub_list(sl,l):
	    results=[]
	    sll=len(sl)
	    for ind in (i for i,e in enumerate(l) if e==sl[0]):
	        if l[ind:ind+sll]==sl:
	            results.append((ind,ind+sll-1))

	    return results

	for e in range(epoch):
		
		#debug用early stop參數
		acc = 0
		cnt = 0
		running_loss = 0.0
		
		#DRCD
		for d in jf['data']:
			for p in d['paragraphs']:

				for q in p['qas']:

					for a in q['answers']:

						question = q['question']
						text = p['context']

						if len(text)>450:
							text = text[:450]

						encoding = tokenizer.encode_plus(question, text)
						input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
						all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
						c = input_ids
						b = tokenizer.encode(a['text'])
						b = b[1:-1]
						res = find_sub_list(b, c)

						if res != []:
							start_position = res[0][0]
							end_position = res[0][1]
						else:
							start_position = 0
							end_position = 0

						# 將參數梯度歸零
						optimizer.zero_grad()
						y = ''.join(all_tokens[start_position : end_position+1])
						print(a['text'], y)
						if y == a['text']:
							# forward pass
							outputs = model(torch.tensor([input_ids]).to(device) , token_type_ids=torch.tensor([token_type_ids]).to(device) , start_positions = torch.tensor([start_position]).to(device) , end_positions = torch.tensor([end_position]).to(device) )
							cnt+=1
							loss = outputs[0]
							# backward
							loss.backward()
							optimizer.step()
		
							running_loss += loss.item()
		torch.save(model, outputfile)
		#FGCB
		ix = 0
		for d in q_jfB:

			for q in d['QUESTIONS']:

				question = q['QTEXT'][2:]
				text = d['DTEXT']

				if len(text)>450:
					text = text[:450]

				encoding = tokenizer.encode_plus(question, text)
				input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
				all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
				c = input_ids


				for multia in ans_lstB[ix].split(';'):

					b = tokenizer.encode(multia)
					b = b[1:-1]
					res = find_sub_list(b, c)

					if res != []:
						start_position = res[0][0]
						end_position = res[0][1]
						break
					else:
						start_position = 0
						end_position = 0

				optimizer.zero_grad()
				y = ''.join(all_tokens[start_position : end_position+1])
				print(multia, y)
				if y == multia: 
					# forward pass
					outputs = model(torch.tensor([input_ids]).to(device) , token_type_ids=torch.tensor([token_type_ids]).to(device) , start_positions = torch.tensor([start_position]).to(device) , end_positions = torch.tensor([end_position]).to(device) )
					cnt+=1
					loss = outputs[0]
					# backward
					loss.backward()
					optimizer.step()
					running_loss += loss.item()
				ix+=1	
		torch.save(model, outputfile)
		#FGCA
		ix = 0
		for d in q_jfB:

			for q in d['QUESTIONS']:

				question = q['QTEXT'][2:]
				text = d['DTEXT']

				if len(text)>450:
					text = text[:450]


				encoding = tokenizer.encode_plus(question, text)
				input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
				all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
				c = input_ids

				res = find_sub_list(b, c)

				for multia in ans_lstB[ix].split(';'):

					b = tokenizer.encode(multia)
					b = b[1:-1]
					res = find_sub_list(b, c)

					if res != []:
						start_position = res[0][0]
						end_position = res[0][1]
						break
					else:
						start_position = 0
						end_position = 0

				optimizer.zero_grad()
				y = ''.join(all_tokens[start_position : end_position+1])
				print(multia, y)
				if y == multia: 
					# forward pass
					outputs = model(torch.tensor([input_ids]).to(device) , token_type_ids=torch.tensor([token_type_ids]).to(device) , start_positions = torch.tensor([start_position]).to(device) , end_positions = torch.tensor([end_position]).to(device) )
					cnt+=1
					loss = outputs[0]
					# backward
					loss.backward()
					optimizer.step()
					running_loss += loss.item()
				ix+=1	
		torch.save(model, outputfile)
		# #OUR QA SET
		# for d in our_jf:

		# 	for q in d['QUESTIONS']:

		# 		question = q['QTEXT'][2:]
		# 		text = d['DTEXT']

		# 		if len(text)>470:
		# 			text = text[:470]

		# 		encoding = tokenizer.encode_plus(question, text)
		# 		input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
		# 		all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
		# 		c = input_ids
				
		# 		for multia in q['ANSWER'].split(';'):
		# 			b = tokenizer.encode(multia)
		# 			b = b[1:-1]
		# 			res = find_sub_list(b, c)

		# 			if res != []:
		# 				start_position = res[0][0]
		# 				end_position = res[0][1]
		# 				break
		# 			else:
		# 				start_position = 0
		# 				end_position = 0

		# 		optimizer.zero_grad()
		# 		y = ''.join(all_tokens[start_position : end_position+1])
		# 		print(multia, y)
		# 		if y == multia: 
		# 			# forward pass
		# 			outputs = model(torch.tensor([input_ids]).to(device) , token_type_ids=torch.tensor([token_type_ids]).to(device) , start_positions = torch.tensor([start_position]).to(device) , end_positions = torch.tensor([end_position]).to(device) )
		# 			cnt+=1
		# 			loss = outputs[0]
		# 			# backward
		# 			loss.backward()
		# 			optimizer.step()
		# 			running_loss += loss.item()


		if cnt!= 0:
			print(e, ": cnt=\t", cnt, "avg_loss =\t", running_loss/cnt)

		torch.save(model, outputfile)

if __name__ == "__main__":
	main(sys.argv[1:])
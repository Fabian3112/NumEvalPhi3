
from datasets import Dataset
import json

def generate_qqa_cot_dataset(path,tokenizer):
    
    system_message = 'you are an AI assistant trained to answere Questions. You will decide wether " Option 1 " or " Option 2 " is correct. Think step by step.'

    with open(path,'r') as f:
        data = json.load(f)

    with open('chaine_of_thought_examples.json','r') as f:
        examples = json.load(f)

    data_dict = {'inputs':[],'text':[],'labels':[]}

    for item in data:
        
        
        template = '{question} Option 1 : {Option1} or Option 2 : {Option2}. lets think step by step.'
        
        messages = [{"role":"system", "content": " " + system_message}]
        for example in examples:
            question = template.format(question=example['question'],Option1=example['Option1'],Option2=example['Option2'])
            messages.append({"role": "user", "content": " " + question})
            messages.append({"role": "assistant", "content": " " + example['answer']})
            #example_text += question + '\n'

        

        question = template.format(question=item['question'],Option1=item['Option1'],Option2=item['Option2'])
        messages.append({"role": "user", "content": " " + question})

        input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        data_dict["inputs"].append(input)
        
        if 'gpt_answers' in item:
            messages.append({"role": "assistant", "content": item['gpt_answers']})
        else:
            print('not for training')
            print(item)
            messages.append({"role": "assistant", "content": item['answer']})
            
        input = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        data_dict["text"].append(input)
        data_dict["labels"].append(item['answer'])
    
    return  Dataset.from_dict(data_dict)

def generate_qqa_dataset(path,tokenizer):
    system_message = 'you are an AI assistant trained to answere Questions. You will answere with either " Option 1 " or " Option 2 "'

    with open(path,'r') as f:
        data = json.load(f)


    data_dict = {'text':[],'inputs':[],'labels':[]}
    for item in data:
        answers = ['Option 1','Option 2']
        if '1' in item["answer"]:
            answer = answers[0]
            answer_reversed = answers[1]
        else:
            answer = answers[1]
            answer_reversed = answers[0]
            

        question = item['question'] + ' Option 1 : ' + item['Option1'] + ' Option 2 : ' + item['Option2']
        messages = [{"role":"system", "content": " " + system_message },{"role": "user", "content": " " + question},{"role": "assistant", "content": " " + answer}]
        input = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        data_dict['text'].append(input)
        
        # add inout and label
        messages = [{"role":"system", "content": " " + system_message },{"role": "user", "content": " " + question}]
        input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        data_dict['inputs'].append(input)
        data_dict['labels'].append(answer)


        # add same question reversed Options
        question = item['question'] + ' Option 1 : ' + item['Option2'] + ' Option 2 : ' + item['Option1']
        messages = [{"role":"system", "content":" " + system_message },{"role": "user", "content": " " + question},{"role": "assistant", "content": " " + answer_reversed}]
        input = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        data_dict['text'].append(input)

        # add inout and label
        messages = [{"role":"system", "content":" " + system_message },{"role": "user", "content": " " + question}]
        input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        data_dict['inputs'].append(input)
        data_dict['labels'].append(answer_reversed)


    return  Dataset.from_dict(data_dict)

def remove_key_json(json_data, key_to_remove):
    return [{key: value for key, value in data.items() if key not in key_to_remove} for data in json_data]


def gerernate_qqp(path,tokenizer):
    with open(path,'r') as f:
        data = json.load(f)

    data = remove_key_json(data, ['length', 'offset', 'comment_sci_10E_char', 
                                    'comment_sci_10E', 'title_sci_10E_char', 
                                    'title_sci_10E', 'UNIQUE_STORY_INDEX','comment_char'])

    keys = data[0].keys()
    data_dict = {}
    for key in keys:
        data_dict[key] = []

    data_dict['text'] = []
    data_dict['inputs'] = []
    data_dict['labels'] = []

    for i,item in enumerate(data[:1000]):
        print(i/len(data) * 100 )
        for key in keys:
            data_dict[key].append(str(item[key]))

        examples = {
            'comments' : [
                "COSTCO WHOLESALE <COST.O> UP [Num] PCT IN PREMARKET TRADING AFTER  RESULTS",
            #    "BRIEF-Henan Shuanghui Investment's H1 net profit up [Num] pct",
                "EASTERN TREADS LTD - SEPT QTR NET SALES 217 MLN RUPEES VS [Num] MLN  RUPEES YR AGO",
            #    "NSEI Block Deal Ultratech Cement 50400 shares at [Num]0 INR <ULTC.NS>",
            #    "NYSE ORDER IMBALANCE <SWN.N> [Num] SHARES ON BUY SIDE",
                "NYSE INDICATION <BRKa.N> LAST 132295.0 BID 131000.0 ASK [Num]",
            #    "This is a TEST snap for ECONTEST2=ECI ([Num]), Please ignore."
            ],
            'magnitudes' :[0,2,6] #[0,1,2,3,4,5,6,7]
        }
        messages = []
        for i,comment in enumerate(examples["comments"]):
            messages.append({"role":"user", "content":f'Predict the magnitude for [Num] in the comment: {comment}'})
            messages.append({"role": "assistant", "content": f" magnitude : {examples['magnitudes'][i]}"})

        messages.append({"role":"user", "content":f'Predict the magnitude for [Num] in the comment: {item["comment"]}'})
        input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        data_dict["inputs"].append(input)

        messages.append({"role": "assistant", "content": f" magnitude : {item['magnitude']}"})
        input = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        data_dict["text"].append(input)
        data_dict["labels"].append(f"magnitude : {item['magnitude']}")

    return Dataset.from_dict(data_dict)


#eval(model,tokenizer,0,test_dataset,'test',out_file,out_predictions_file)
def eval(model,tokenizer,epoch,data,mode,out_file,out_predictions_file):

    classes = {}

    true_count = 0
    total_count = 0

    predictions = []
    labels = []

    for i,input in enumerate(data["inputs"]):
        total_count += 1
        answer = data["labels"][i]
        input = tokenizer.encode(input,return_tensors="pt")
        input = input.to('mps')
        outputs = model.generate(input, max_new_tokens=32)
        text = tokenizer.batch_decode(outputs)[0]

        predicted_answer = text.split("<|assistant|>")[-1]

        if not answer in classes:
            classes[answer] = {'TP':0,'FP':0,'FN':0}

        if answer in predicted_answer:
            true_count += 1
            classes[answer]['TP'] += 1
        
        else:
            classes[answer]['FN'] += 1

            if not predicted_answer in classes:
                classes[predicted_answer] = {'TP':0,'FP':0,'FN':0}

            classes[predicted_answer]['FP'] += 1


        predictions.append(text.split("<|assistant|>")[-1])
        labels.append(answer)
        #else:
        #    print(f"worng predicted {text.split("<|assistant|>")[-1]} but was {answer}")

        print(mode)    
        print(f'{total_count / len(data["inputs"]) *100}% : {true_count/total_count *100}%')

    with open(out_file,'a') as f:
        f.write(f'epoch {epoch} : {mode} \n')
        
        f.write(f'accuracy: {true_count/total_count *100} % \n')
        f.write(f'totalcount: {total_count} \n')
        f.write(f'Klass predictions: {classes} \n')

        f.write('-----------------------------\n')

    with open(out_predictions_file,'a') as f:
        f.write(f'\n epoch {epoch} : {mode}')
        f.write('-----------------------------\n')
        f.write(f'predictions : {predictions}')
        f.write(f'labels : {labels}')
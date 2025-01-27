import argparse
import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


def load_plm(model_path='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_path,device_map="auto",)
    print("Load Model:", model_path.split('/')[-1])
    model = AutoModel.from_pretrained(model_path,device_map="auto",torch_dtype=torch.float16)
    return tokenizer, model


def load_data(args):
    # 需要传入的是一个prompt文件夹args.root，文件夹下面是txt文件，每个txt文件的文件名是prompt的id
    id2prompt = {}
    for file in os.listdir(os.path.join(args.root, 'prompts')):
        with open(os.path.join(args.root, 'prompts', file), 'r') as f:
            if file.endswith('.txt'):
                data = f.read()
                if data:
                    id2prompt[file.split('.')[0]] = data
                else:
                    print("Empty prompt: ", file.split('.')[0])
    print('Prompt number: ', len(id2prompt))
    return id2prompt

def generate_location_embedding(args, loc_prompt: dict, tokenizer, model):
    print(f'Generate Text Embedding: ')
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(loc_prompt)):
            if i % 100 == 0:
                print(f'Processing {i}...')
            prompt = loc_prompt[str(i)]
            encoded_sentences = tokenizer(prompt, max_length=args.max_sent_len,
                                        truncation=True, return_tensors='pt',padding="longest").to("cuda")
            outputs = model(input_ids=encoded_sentences.input_ids,
                            attention_mask=encoded_sentences.attention_mask)
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            
            embeddings.append(mean_output)

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    file = os.path.join(args.root,  'location.emb-' + args.llm_path.split('/')[-1] + "-td" + ".npy")
    np.save(file, embeddings)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="../data")
    parser.add_argument('--gpu_id', type=str, default='1,2,4,6,7', help='gpu id')
    parser.add_argument('--llm_path', type=str,
                        default="path to Llama-3.2-1B-Instruct")
    parser.add_argument('--max_sent_len', type=int, default=4096)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    loc_prompt = load_data(args)

    plm_tokenizer, plm_model = load_plm(args.llm_path)
    if plm_tokenizer.pad_token_id is None:
        plm_tokenizer.pad_token_id = 0

    generate_location_embedding(args, loc_prompt, plm_tokenizer,
                            plm_model)



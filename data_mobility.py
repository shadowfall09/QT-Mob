import random
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from prompt_mobility import *
import pandas as pd
import pickle
from tqdm import tqdm


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.data_path = args.data_path # 数据路径
        self.max_his_len = args.max_his_len # 最大历史记录长度
        self.his_sep = args.his_sep # The separator used for history
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix # 是否加上序号
        self.sft_json_output = args.sft_json_output # 是否输出json文件
        self.indexing = args.indexing # 是否使用index表示, False表示使用(xxx,xxx)的location表示
        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None
        self.task_prompt = None
        self.data_filename = args.data_filename 
        self.multi_seq = args.multi_seq
        self.add_profile = args.add_profile
        self.multi_rec = args.multi_rec
        self.single_rec = args.single_rec
        self.abalation_location_prompt = args.ablation_location_prompt
               
    def _load_data(self):
        raise NotImplementedError

    def get_all_items(self):
        # 返回所有item的index表示
        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    
    def get_prefix_allowed_tokens_fn(self, tokenizer, test_task):
        # 返回一个函数，该函数返回当前token的allowed_tokens
        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                self.token_len = len(index)
                len_of_token =  len(tokenizer(index[0])["input_ids"])-1
                token_ids = [tokenizer(token)["input_ids"][len_of_token] for token in index]
                if token_ids[0] not in self.allowed_tokens.keys():
                    self.allowed_tokens[token_ids[0]] = set()
                self.allowed_tokens[token_ids[0]].add(token_ids[1])
                for i in range(2, len(token_ids)):
                    if tuple(token_ids[0:i]) not in self.allowed_tokens.keys():
                        self.allowed_tokens[tuple(token_ids[0:i])] = set()
                    self.allowed_tokens[tuple(token_ids[0:i])].add(token_ids[i])
            for index in self.indices.values():
                for i, token in enumerate(index): # i表示token的位置,取值范围为0-n，token就n+1位
                    token_id = tokenizer(token)["input_ids"][len_of_token]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
                
        if test_task == "seq":
            sep = tokenizer(" will visit POI index ",add_special_tokens=False)["input_ids"][1:]
        elif test_task == "recovery":
            sep = tokenizer(" visited POI index ",add_special_tokens=False)["input_ids"][1:]
        
        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            # print(tokenizer.decode(sentence))
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    if i == self.token_len:
                        return [tokenizer.eos_token_id]
                    if i == 0 or reversed_sent[0]<20:
                        return list(self.allowed_tokens[i])
                    if i == 1:
                        return list(self.allowed_tokens[reversed_sent[0]])
                    return list(self.allowed_tokens[tuple(reversed_sent[0:i][::-1])])
            print("Warning: sep not found")

        return prefix_allowed_tokens_fn

    def _process_data(self):
        raise NotImplementedError    
    
    def set_prompt(self, prompt_id):
        self.test_prompt_id = prompt_id

    def __len__(self):
        return len(self.inter_data)
    
    def _get_text_data(self, data, prompt, sft_format=False):
        if self.indexing:
            sys_prompt = system_prompt
        else:
            sys_prompt = system_prompt_not_indexing.format(max_poi=len(self.indices)-1)
        instruction = sys_prompt + self.task_prompt + prompt.format(**data)
        response = data["response"]
        prediction = data["prediction"] if "prediction" in data else ""

        if self.mode == 'test':
            input = sft_prompt.format(instruction = instruction, response = response, prediction = "")
            return input, prediction
        
        if sft_format:
            input = sft_prompt.format(instruction = instruction, response = "", prediction = "")
            output = sft_prompt.format(instruction = instruction, response = response, prediction = prediction)
        else:
            input = instruction
            output = response + prediction
        return input, output
    
    def __getitem__(self, index):
        d = self.inter_data[index]
        if self.mode == 'test':
            prompt_id = self.test_prompt_id # 测试时使用指定的prompt
        else:
            prompt_id = random.randint(0, len(self.prompts) - 1) # 随机选择一个prompt

        prompt = self.prompts[prompt_id] # 获取prompt
        input, output = self._get_text_data(d, prompt, not self.sft_json_output)
        return dict(input_ids=input, labels=output)



class SeqDataset(BaseDataset):
    # Task -- Next Location Prediction

    def __init__(self, args, mode="train"):
        super().__init__(args)

        self.mode = mode # train, valid, test
        
        self.prompts = all_prompt["seq"] # 所有的prompt
        self.task_prompt = task_prompt
        
        print("Dataset Name: ", self.mode, "SeqDataset")

        self._load_data()
        self._remap_items()
        self.inter_data = self._process_data()
        
        print("Total number of data: ", len(self.inter_data))


    def _load_data(self):
        # load data
        self.inter_data = pd.read_csv(os.path.join(self.data_path, self.data_filename[:-4]+"_"+self.mode+".csv"), converters={'trips':eval})
        with open(os.path.join(self.data_path, "user_index.json"), 'r') as f:
            self.user_index = json.load(f)
        
        # 读取index文件
        with open(os.path.join(self.data_path, self.index_file), 'r') as f:
            self.indices = json.load(f)
        if not self.indexing:
            self.indices = {k: [f"{k}"] for k in self.indices.keys()}
            
        with open(os.path.join(self.data_path, "loc2id"), 'rb') as file:
            self.loc2id = pickle.load(file)
        
        self.user_profile = pd.read_csv(os.path.join(self.data_path, "user_profile.csv"), converters={'latest_5_trips': eval},sep="|")

    # trajectory: [(index, time, loc[0], loc[1], user_id, traj_id), ...]
    def _remap_items(self):
        all_trajectory = []
        for idx,row in self.inter_data.iterrows():
            user_id = str(row['user_id'])
            traj_id = str(row['traj_id'])
            updates = [i.split(',') for i in row['trips']]
            locs = [(float(i[0]), float(i[1]),i[2], user_id, traj_id) for i in updates]
            all_trajectory.append(locs)
        # item转换成index表示
        self.remapped_inters = []
        for trajectory in all_trajectory:
            new_trajectory = [("".join(self.indices[str(self.loc2id[(loc[0],loc[1])])]),loc[2],loc[0],loc[1],loc[3],loc[4]) for loc in trajectory]
            self.remapped_inters.append(new_trajectory)

    def _process_data(self):

        inter_data = []
        for trajectory in tqdm(self.remapped_inters):
            if self.multi_seq and self.mode != "test":
                start = 2
            else:
                start = len(trajectory)-1
            for i in range(start, len(trajectory)):
                one_data = dict()
                one_data["user"] = trajectory[i][4]
                one_data["response"] = "At time " + trajectory[i][1] + ", user "+trajectory[i][4]+" will visit POI index "
                one_data["prediction"] = trajectory[i][0]
                one_data["time"] = trajectory[i][1]
                history = trajectory[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]# 只保留最近的max_his_len个历史记录
                history = ["At time " + item_idx[1] + ", user " + item_idx[4] + " visited POI index " + item_idx[0] + "." for item_idx in history]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)] # 添加序号前缀 1. item1 
                one_data["inters"] = self.his_sep.join(history)
                if self.add_profile:
                    profile = self.user_profile.loc[self.user_profile['user_id'] == int(trajectory[i][4])]
                    one_data["profile"] = "User "+trajectory[i][4]+" has the following profile: "+profile['prompt'].values[0]+" "
                else:
                    one_data["profile"] = ""
                inter_data.append(one_data)
        return inter_data


class RecoveryDataset(BaseDataset):
    # Task -- Trajectory Recovery --10 Prompt
    # 有训练集，验证集和测试集

    def __init__(self, args, mode="train"):
        super().__init__(args)

        self.mode = mode # train, valid, test
        
        self.prompts = all_prompt["recovery"] # 所有的prompt
        self.task_prompt = task_prompt
        
        print("Dataset Name: ", self.mode, "RecoveryDataset")

        self._load_data()
        self._remap_items()
        self.inter_data = self._process_data()
        
        print("Total number of data: ", len(self.inter_data))


    def _load_data(self):
        # load data
        self.inter_data = pd.read_csv(os.path.join(self.data_path, self.data_filename[:-4]+"_"+self.mode+".csv"), converters={'trips':eval})
        with open(os.path.join(self.data_path, "user_index.json"), 'r') as f:
            self.user_index = json.load(f)    
                    
        # 读取index文件
        with open(os.path.join(self.data_path, self.index_file), 'r') as f:
            self.indices = json.load(f)
        if not self.indexing:
            self.indices = {k: [f"{k}"] for k in self.indices.keys()}
            
        with open(os.path.join(self.data_path, "loc2id"), 'rb') as file:
            self.loc2id = pickle.load(file)
        
        self.user_profile = pd.read_csv(os.path.join(self.data_path, "user_profile.csv"), converters={'latest_5_trips': eval},sep="|")



    def _remap_items(self):
        all_trajectory = []
        for idx,row in self.inter_data.iterrows():
            user_id = str(row['user_id'])
            updates = [i.split(',') for i in row['trips']]
            traj_id = str(row['traj_id'])
            locs = [(float(i[0]), float(i[1]),i[2],user_id, traj_id) for i in updates]
            all_trajectory.append(locs)
        # item转换成index表示
        self.remapped_inters = []
        for trajectory in all_trajectory:
            new_trajectory = [("".join(self.indices[str(self.loc2id[(loc[0],loc[1])])]),loc[2],loc[0],loc[1],loc[3],loc[4]) for loc in trajectory]
            self.remapped_inters.append(new_trajectory)

    def _process_data(self):
        
        def generate_multi_mask(history):
            one_data = dict()
            one_data["user"] = history[-1][4]
            one_data["response"] = ""
            mask_count = random.randint(max(1, int(0.2 * len(history))), max(1, int(0.5 * len(history)))) # 随机选择20%-50%的位置作为mask
            mask_indices = random.sample(range(1,len(history)), mask_count) # 从第2个到最后一个位置随机选择这些位置作为mask
            if self.mode != "test":
                one_data["prediction"] = self.his_sep.join(["At time " + item_idx[1] + ", user " + item_idx[4] + " visited POI index " + item_idx[0] + "." for item_idx in history])
            else:
                one_data["prediction"] = [{"answer": "At time " + item_idx[1] + ", user " + item_idx[4] + " visited POI index " + item_idx[0] + ".", "mask": idx in mask_indices} for idx, item_idx in enumerate(history)]
            
            for mask_idx in mask_indices:
                history[mask_idx] = ("[MASK]", history[mask_idx][1], history[mask_idx][2], history[mask_idx][3], history[mask_idx][4])
            history = ["At time " + item_idx[1] + ", user " + item_idx[4] + " visited POI index " + item_idx[0] + "." for item_idx in history]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]  # 添加序号前缀 1. item1
            one_data["inters"] = self.his_sep.join(history)
            one_data["multi"] = " and output the complete current trajectory" 
            return one_data
        
        def generate_single_mask(history):
            one_data_list = []
            # one_trips_new = []
            # one_trips_sparse = []
            # one_num_label = []
            mask_count = random.randint(max(1, int(0.2 * len(history))), max(1, int(0.5 * len(history)))) # 随机选择20%-50%的位置作为mask
            mask_indices = random.sample(range(1,len(history)), mask_count) # 从第2个到最后一个位置随机选择这些位置作为mask
            # for i in range(len(history)):
            #     timestamp = pd.to_datetime(history[i][1]).timestamp()
            #     one_trips_new.append((int(history[i][0]), history[i][2], history[i][3], timestamp))
            #     if i not in mask_indices:
            #         one_num_label.append(0)
            #         one_trips_sparse.append((int(history[i][0]), history[i][2], history[i][3], timestamp))
            #     else:
            #         one_num_label[-1] += 1
            for mask_idx in mask_indices:
                one_data = dict()
                history_one = history.copy()
                one_data["user"] = history_one[mask_idx][4]
                one_data["response"] = "At time " + history_one[mask_idx][1] + ", user " + history_one[mask_idx][4] + " visited POI index "
                one_data["prediction"] = history_one[mask_idx][0]
                history_one = [("At time " + item_idx[1] + ", user " + item_idx[4] + " visited POI index " + ("[MASK]" if idx == mask_idx else "[UNKNOWN]" if idx in mask_indices else item_idx[0]) + ".") for idx, item_idx in enumerate(history_one)]
                if self.add_prefix:
                    history_one = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history_one)]
                one_data["inters"] = self.his_sep.join(history_one)
                one_data["multi"] = ""
                one_data_list.append(one_data)
            return one_data_list
            # return one_trips_new, one_trips_sparse, one_num_label
                    

        inter_data = []
        # trips_new = [] # 用于baseline的复现
        # trips_sparse = [] # 用于baseline的复现
        # num_labels = [] # 用于baseline的复现
        # user_list = [] # 用于baseline的复现
        for trajectory in tqdm(self.remapped_inters):
            history = trajectory
            if self.max_his_len > 0:
                history = history[:self.max_his_len]  # 只保留最近的max_his_len个历史记录
            if self.multi_rec and self.mode != "test":
                one_data = generate_multi_mask(history.copy())
                inter_data.append(one_data)
            if self.single_rec or self.mode == "test":
                one_data_list = generate_single_mask(history.copy())
                inter_data.extend(one_data_list)
            # one_trips_new, one_trips_sparse, one_num_label = generate_single_mask(history)
            # trips_new.append(one_trips_new)
            # user_list.append(trajectory[0][4])
            # trips_sparse.append(one_trips_sparse)
            # num_labels.append(one_num_label)
        if self.add_profile:
            for one_data in inter_data:
                profile = self.user_profile[self.user_profile['user_id']==int(one_data["user"])]
                one_data["profile"] = "User "+one_data["user"]+" has the following profile: "+profile['prompt'].values[0]+" "
        else:
            for one_data in inter_data:
                one_data["profile"] = "" 
        # df = pd.DataFrame({
        #     'trips_new': trips_new,
        #     'trips_sparse': trips_sparse,
        #     'num_labels': num_labels,
        #     'user_list': user_list
        # })
        # df.to_csv(os.path.join(self.data_path, f"{self.mode}_recovery_data.csv"), index=False)
        return inter_data    


class Index2LocationDataset(BaseDataset):
    # Task -- Index to Location

    def __init__(self, args):
        super().__init__(args)
        
        self.prompts = all_prompt["index"] # 所有的prompt
        self.task_prompt = POI_prompt
        self.mode = "train"
        
        print("Dataset Name: Index2LocationDataset")

        self._load_data()
        self.inter_data = self._process_data()
        
        print("Total number of data: ", len(self.inter_data))


    def _load_data(self):
        # load data
        location_prompt = {}
        for file in os.listdir(os.path.join(self.data_path, "prompts")): 
            with open(os.path.join(self.data_path, "prompts", file), 'r') as f:
                content = f.read().split("\n")
                if self.abalation_location_prompt=="1":
                    content.pop(3)
                elif self.abalation_location_prompt=="2":
                    content.pop(4)
                elif self.abalation_location_prompt=="3":
                    content.pop(5)
                content = "\n".join(content)                
                location_prompt[file.split(".")[0]] = content
        self.location_prompt = location_prompt
        
        # 读取index文件
        with open(os.path.join(self.data_path, self.index_file), 'r') as f:
            self.indices = json.load(f)
        if not self.indexing:
            self.indices = {k: [f"{k}"] for k in self.indices.keys()}
            
        assert len(self.indices) == len(self.location_prompt) , "The number of indices and prompts should be the same."
        # 会有一模一样的location


    def _process_data(self):
        inter_data = []
        for index, prompt in self.location_prompt.items():
            one_data = dict()
            one_data["index"] = "".join(self.indices[index])
            one_data["response"] = prompt
            inter_data.append(one_data)
        return inter_data
    

class Location2IndexDataset(BaseDataset):
    # Task -- Location to Index

    def __init__(self, args):
        super().__init__(args)
        
        self.prompts = all_prompt["location"] # 所有的prompt
        self.task_prompt = POI_prompt
        self.mode = "train"

        print("Dataset Name: Location2IndexDataset")

        self._load_data()
        self.inter_data = self._process_data()
        
        print("Total number of data: ", len(self.inter_data))


    def _load_data(self):
        # load data
        location_prompt = {}
        for file in os.listdir(os.path.join(self.data_path, "prompts")): 
            with open(os.path.join(self.data_path, "prompts", file), 'r') as f:
                content = f.read().split("\n")
                if self.abalation_location_prompt=="1":
                    content.pop(3)
                elif self.abalation_location_prompt=="2":
                    content.pop(4)
                elif self.abalation_location_prompt=="3":
                    content.pop(5)
                content = "\n".join(content)
                location_prompt[file.split(".")[0]] = content
        self.location_prompt = location_prompt
        
        # 读取index文件
        with open(os.path.join(self.data_path, self.index_file), 'r') as f:
            self.indices = json.load(f)
        if not self.indexing:
            self.indices = {k: [f"{k}"] for k in self.indices.keys()}
            
        assert len(self.indices) == len(self.location_prompt) , "The number of indices and prompts should be the same."
        # 会有一模一样的location


    def _process_data(self):
        inter_data = []
        for index, prompt in self.location_prompt.items():
            one_data = dict()
            one_data["response"] = "".join(self.indices[index])
            one_data["location"] = prompt
            inter_data.append(one_data)
        return inter_data
    
    
class TrajectoryTranslationDataset(BaseDataset):
    # Task -- Trajectory Translation

    def __init__(self, args, mode="train"):
        super().__init__(args)

        self.mode = mode
        
        self.prompts = all_prompt["trans"]
        self.task_prompt = task_prompt
        
        print("Dataset Name: ", self.mode, "TrajectoryTranslationDataset")

        self._load_data()
        self._remap_items()
        self.inter_data = self._process_data()
        
        print("Total number of data: ", len(self.inter_data))


    def _load_data(self):
        # load data
        self.inter_data = pd.read_csv(os.path.join(self.data_path, self.data_filename[:-4]+"_"+self.mode+".csv"), converters={'trips':eval})
        
        location_prompt = {}
        for file in os.listdir(os.path.join(self.data_path, "prompts")): 
            with open(os.path.join(self.data_path, "prompts", file), 'r') as f:
                content = f.read().split("\n")
                if self.abalation_location_prompt=="1":
                    content.pop(3)
                elif self.abalation_location_prompt=="2":
                    content.pop(4)
                elif self.abalation_location_prompt=="3":
                    content.pop(5)
                content = "\n".join(content)
                location_prompt[file.split(".")[0]] = content
        self.location_prompt = location_prompt
        
        # 读取index文件
        with open(os.path.join(self.data_path, self.index_file), 'r') as f:
            self.indices = json.load(f)
        if not self.indexing:
            self.indices = {k: [f"{k}"] for k in self.indices.keys()}
            
        with open(os.path.join(self.data_path, "loc2id"), 'rb') as file:
            self.loc2id = pickle.load(file)

    # trajectory: [(index, time, loc[0], loc[1], user_id, traj_id), ...]
    def _remap_items(self):
        all_trajectory = []
        for idx,row in self.inter_data.iterrows():
            user_id = str(row['user_id'])
            traj_id = str(row['traj_id'])
            updates = [i.split(',') for i in row['trips']]
            locs = [(float(i[0]), float(i[1]),i[2], user_id, traj_id) for i in updates]
            all_trajectory.append(locs)
        # item转换成index表示
        self.remapped_inters = []
        for trajectory in all_trajectory:
            new_trajectory = [("".join(self.indices[str(self.loc2id[(loc[0],loc[1])])]),loc[2],loc[0],loc[1],loc[3],loc[4]) for loc in trajectory]
            self.remapped_inters.append(new_trajectory)


    def _process_data(self):
        inter_data = []
        for trajectory in tqdm(self.remapped_inters):
            history = trajectory
            if self.max_his_len > 0:
                history = history[:self.max_his_len]  # 只保留最近的max_his_len个历史记录
            one_data = dict()
            one_data["user"] = trajectory[0][4]
            one_data["response"] = self.his_sep.join(["[ "+str(k+1) + " ] At time " + item_idx[1] + ", user " + item_idx[4] + " visited POI index " + item_idx[0] + "." for k, item_idx in enumerate(history)])
            one_data["inters"] = ["[ "+str(k+1) + " ]\nTime: " + item_idx[1] + "\nLocation Description: "+ "\n".join(self.location_prompt[str(self.loc2id[(item_idx[2],item_idx[3])])].split("\n")[:3]) for k, item_idx in enumerate(history)]
            one_data["inters"] = "\t\n".join(one_data["inters"])
            inter_data.append(one_data)
        return inter_data
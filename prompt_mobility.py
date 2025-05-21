# alpaca style template
# sft_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request." \
#              "\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}{prediction}"
             
sft_prompt = "[INST] {instruction} [/INST] {response}{prediction}"

system_prompt = """\
<<SYS>> You are a helpful assistant that predicts human mobility trajectories in a city. <<\SYS>> \
Each POI is encoded into a 4-token-length index called "POI index", which contains its spatial information.
"""

system_prompt_not_indexing = """\
<<SYS>> You are a helpful assistant that predicts human mobility trajectories in a city. <</SYS>> \
Each "POI index" is an integer starting from 0 to {max_poi}.
"""

POI_prompt = """\
Your goal is to learn the spatial and locational information represented by each POI index.
Question: """

# The notation [d: 5.2 km, t: 3.5 h] between POIs represents the straight-line distance and time interval. 

task_prompt = """\
A trajectory is a time-ordered sequence of POI indices, where each POI visit is noted by the index and visit time. Each visit reflects a unique purpose related to that location.
Task: """

user_history_prompt = """User {user} had the following HISTORICAL trajectories: """

all_prompt = {}


# =====================================================
# Task 1 -- Next POI Prediction -- 10 Prompt
# =====================================================
seq_prompt = []

prompt = "{profile}The following data is a trajectory of user {user}. {inters} Given the data, At {time}, Which POI index will user {user} visit?"
seq_prompt.append(prompt)

# ###--1
# prompt = "A user has visited the following POI indices in sequence: {inters} What is the most likely next POI index the user will visit at {time}?"
# seq_prompt.append(prompt)

# ###--2
# prompt = "Given a user's movement path across these POI indices: {inters} Predict the next POI index they are likely to visit at {time}."
# seq_prompt.append(prompt)

# ###--3
# prompt = "Based on the user's travel history: {inters} What is the next probable POI index in their trajectory at {time}?"
# seq_prompt.append(prompt)

# ###--4
# prompt = "A user's POI index history is as follows: {inters} Can you infer the next POI index they are most likely to go to at {time}?"
# seq_prompt.append(prompt)

# ###--5
# prompt = "After visiting these POI indices in the following order: {inters} Which POI index is the user likely to visit next at {time}?"
# seq_prompt.append(prompt)

# ###--6
# prompt = "A user has traveled to POI indices one after another. {inters} What is the next POI index they might choose at {time}?"
# seq_prompt.append(prompt)

# ###--7
# prompt = "Using the user's visited POI indices in sequence: {inters} Predict the next POI index the user is expected to visit at {time}."
# seq_prompt.append(prompt)

# ###--8
# prompt = "Based on the user's sequence of visited POI indices: {inters} What would be the most probable next POI index at {time}?"
# seq_prompt.append(prompt)

# ###--9
# prompt = "The user's journey so far includes the followibg visits. {inters} Where might he go next based on this trajectory at {time}?"
# seq_prompt.append(prompt)

# ###--10
# prompt = "With the user's recent visits: {inters} Can you predict the next POI index he is most likely to visit at {time}?"
# seq_prompt.append(prompt)

all_prompt["seq"] = seq_prompt

# ========================================================
# Task 2 -- Trajectory Recovery --10 Prompt
# ========================================================
# Remove periods when inputting
recovery_prompt = []

###--1
prompt = "{profile}Given the trajectory consisting of POI indices: {inters} Each [MASK] denotes one missing POI. [UNKNOWN] denotes an unknown POI. Can you recover the missing POI index(not the unknown POI){multi}?"
recovery_prompt.append(prompt)

###--2
prompt = "{profile}The following trajectory has missing POI indices, each marked as [MASK]: {inters} [UNKNOWN] denotes an unknown POI. Can you recover the missing POI index(not the unknown POI){multi}?"
recovery_prompt.append(prompt)

###--3
prompt = "{profile}In the trajectory: {inters} Each [MASK] represents one missing POI and each [UNKNOWN] represents an unknown POI. Can you determine what the missing POI index(not the unknown POI) is{multi}?"
recovery_prompt.append(prompt)

###--4
prompt = "{profile}Given the sequence of POI indices: {inters} The [MASK] denotes a missing POI and [UNKNOWN] denotes an unknown POI, can you fill in the missing POI index(not the unknown POI){multi}?"
recovery_prompt.append(prompt)

###--5
prompt = "{profile}This trajectory: {inters} It includes missing POI index, each indicated by [MASK], and unknown POI index, indicated by [UNKNOWN]. Can you recover the missing POI index(not the unknown POI){multi}?"
recovery_prompt.append(prompt)

###--6
prompt = "{profile}{inters} Each [MASK] in this trajectory marks a missing POI index and each [UNKNOWN] marks an unknown POI index. Can you infer the missing POI index(not the unknown POI){multi}?"
recovery_prompt.append(prompt)

###--7
prompt = "{profile}A user's path is missing specific POI marked by [MASK]: {inters} [UNKNOWN] denotes an unknown POI. Can you predict and recover the missing POI index(not the unknown POI){multi}?"
recovery_prompt.append(prompt)

###--8
prompt = "{profile}In the given sequence: {inters} Each [MASK] indicates one missing POI and each [UNKNOWN] indicates one unknown POI. Can you identify the missing POI index in the trajectory(not the unknown POI){multi}?"
recovery_prompt.append(prompt)

###--9
prompt = "{profile}{inters} Each [MASK] in this trajectory represents a missing POI and each [UNKNOWN] represents an unknown POI. Can you recover the missing point(not the unknown POI){multi}?"
recovery_prompt.append(prompt)

###--10
prompt = "{profile}The following trajectory: {inters} It has missing POI marked by [MASK] and unknown POI marked by [UNKNOWN]. Can you predict the missing POI index which completes the trajectory(not the unknown POI){multi}?"
recovery_prompt.append(prompt)

all_prompt["recovery"] = recovery_prompt


# ========================================================
# Task 3 -- Index to Location -- 8 Prompt
# ========================================================

index2location_prompt = []

#——0
prompt = "Do you know the geographic information of POI index {index}?"
index2location_prompt.append(prompt)

#——1
prompt = "Can you provide the geographical details of POI index {index}?"
index2location_prompt.append(prompt)

#——2
prompt = "Could you tell me the geographical info for POI index {index}?"
index2location_prompt.append(prompt)

#——3
prompt = "What is the geographic information for POI index {index}?"
index2location_prompt.append(prompt)

#——4
prompt = "Please share the geographic data of POI index {index}."
index2location_prompt.append(prompt)

#——5
prompt = "Can you give me details about the geography of POI index {index}?"
index2location_prompt.append(prompt)

#——6
prompt = "Could you explain the geographical characteristics of POI index {index}?"
index2location_prompt.append(prompt)

#——7
prompt = "I'd like to know the geographical information of POI index {index}. Could you provide it?"
index2location_prompt.append(prompt)


all_prompt["index"] = index2location_prompt


# ========================================================
# Task 4 -- Location to Index -- 6 Prompt
# ========================================================

location2index_prompt = []

#——0
prompt = "{location} \n Can you provide its POI index?"
location2index_prompt.append(prompt)

#——1
prompt = "{location} \n Can you tell me its POI index?"
location2index_prompt.append(prompt)

#——2
prompt = "{location} \n What is its POI index?"
location2index_prompt.append(prompt)

#——3
prompt = "{location} \n Please provide its POI index."
location2index_prompt.append(prompt)

#——4
prompt = "{location} \n Can you give me its POI index?"
location2index_prompt.append(prompt)

#——5
prompt = "{location} \n What is the POI index of this location?"
location2index_prompt.append(prompt)


all_prompt["location"] = location2index_prompt


# ========================================================
# Task 5 -- Trajectory Translation
# ========================================================

trajectory_translation_prompt = []

#——0
prompt = """Here's a trajectory description of user {user}:
{inters} 
Can you translate it into a sequence of POI indices?"""
trajectory_translation_prompt.append(prompt)

#——1
prompt = """Given the following user {user} path:
{inters}
Can you convert it into a sequence of POI indices?"""
trajectory_translation_prompt.append(prompt)

#——2
prompt = """User {user}'s path is described as follows:
{inters}
Can you transform it into a sequence of POI indices?"""
trajectory_translation_prompt.append(prompt)


all_prompt["trans"] = trajectory_translation_prompt

import pandas as pd
import numpy as np
import requests
import uuid
import json
from copy import copy, deepcopy
from tqdm import tqdm

from ace import ACE

url = "http://127.0.0.1:1234"
endpoint = "/v1/chat/completions"
api_url = url + endpoint

def call_llm(payload):
    response = requests.post(api_url, json=payload).json()
    output = response["choices"][0]["message"]["content"]
    return output

def edit_playbook(playbook, curator_output):
    #edit playbook
    operations = curator_output["operations"]
    operations

    for elem in operations:
        if elem["type"] == "ADD":

            # Generate a random UUID version 4 (randomly generated)
            uuid_value = str(uuid.uuid4())

            counter = {
                "helpful": 0,
                "harmful": 0,
                "neutral": 0
            }

            content = elem["content"]

            playbook[uuid_value] = {
                "counter": counter,
                "content": content
            }
        
    return playbook
# generate initial playbook
playbook = {}

ace = ACE(playbook, call_llm)

init_prompt = {
    "system_prompt": "You are a helpful assistant that solves problems",
    "user_prompt": """
        Context: the house has 10 doors on the basement floor. it has 6 doors on the upper floor. 
            there are two bedrooms downstairs and two upstairs. 
            there are two bathrooms downstairs and two upstairs. 
            there is a stair connecting the basement and upper floor.
            the master room is on the basement floor. the door from the basement floor opens up the living room.
            the stair is in the living room. 
            there's another door by the kitchen that leads to the living room.
            the main door of the house leads to the living room.
            there is a bedroom in a corner and to the right of the other bedroom upstairs.
            the stairs leads up to a upper living room.
            the stairs is near the ac control box.
            one of the bedrooms is near the ac control box.
            the ac control box is in the center of the upstairs.
        
        Query: tell me the path to my personal bedroom in a linkedlist format starting from the master bedroom where each node is connects to another with an arrow.
    """
}
ground_truth = "Master Bedroom -> Master Bedroom Door -> Living Room -> Stairs -> Upper Living Room -> AC Control Box -> Bedroom Near AC Control Box -> Bedroom to the Right, in the Corner"

for i in tqdm(range(0, 5)):
    print("Iteration: ", i)
    print(json.dumps(playbook, indent=2))

    generator_response = ace._run_generator(init_prompt)
    print(json.dumps(generator_response, indent=2))
    print("---------------------------------------------------------------------------------")

    reflector_response = ace._run_reflector(init_prompt, generator_response, ground_truth)
    print(json.dumps(reflector_response, indent=2))
    print("---------------------------------------------------------------------------------")

    bullets = reflector_response["bullet_tags"]
    # Iterate over each bullet in the list
    for bullet in bullets:
        id = bullet.get("bullet_id")
        tag = bullet.get('tag')  # Get the 'tag' value, or None if not present

        # update playbook tag:
        playbook[id]["counter"][tag] += 1

    curator_response = ace._run_curator(init_prompt, reflector_response)
    print(json.dumps(curator_response, indent=2))

    playbook = edit_playbook(playbook, curator_response)
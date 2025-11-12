from typing import Callable, Optional, Any
import pandas as pd
import numpy as np
import requests
import uuid
import json
from copy import deepcopy

class ACE:
    def __init__(self, playbook, llm_api: Callable[[Any], str]):
        self.playbook = playbook

        self.llm_api = llm_api

        #generator
        self.generator_sys_prompt = """
            You are an analysis expert tasked with answering questions using your knowledge, a curated playbook of strategies and insights and a
            reflection that goes over the diagnosis of all previous mistakes made while answering the question.

            Instructions: - Read the playbook carefully and apply relevant strategies, formulas, and insights - Pay attention to common mistakes
            listed in the playbook and avoid them - Show your reasoning step-by-step - Be concise but thorough in your analysis - If the playbook
            contains relevant code snippets or formulas, use them appropriately - Double-check your calculations and logic before providing the final
            answer

            Your output should be a json object, which contains the following fields: - reasoning: your chain of thought / reasoning / thinking process,
            detailed analysis and calculations - bullet_ids: each line in the playbook has a bullet_id. all bulletpoints in the playbook that’s relevant,
            helpful for you to answer this question, you should include their bullet_id in this list - final_answer: your concise final answer

            Avoid generating any thinking outputs. Just give the straight answer without any thinking.
        """
        self.generator_user_prompt = ""

        #reflector
        self.reflector_sys_prompt = """
            You are an expert analyst and educator. Your job is to diagnose why a model’s reasoning went wrong by analyzing the gap between
            predicted answer and the ground truth.

            Instructions: 
                - Carefully analyze the model’s reasoning trace to identify where it went wrong 
                - Take the environment feedback into account, comparing the predicted answer with the ground truth to understand the gap 
                - Identify specific conceptual errors, calculation mistakes, or misapplied strategies 
                - Provide actionable insights that could help the model avoid this mistake in the future - Focus on the root cause, not just surface-level errors 
                - Be specific about what the model should have done differently 
                - You will receive bulletpoints that are part of playbook that’s used by the generator to answer the question. 
                - You need to analyze these bulletpoints, and give the tag for each bulletpoint, tag can be [‘helpful’, ‘harmful’, ‘neutral’] (for the generator to generate the correct answer)
                - Do not forget to also analyze the desired final output style and compare those styles.

            Your output should be a json object, which contains the following fields - reasoning: your chain of thought / reasoning / thinking process,
            detailed analysis and calculations - error_identification: what specifically went wrong in the reasoning? - root_cause_analysis: why did this
            error occur? What concept was misunderstood? - correct_approach: what should the model have done instead? - key_insight: what
            strategy, formula, or principle should be remembered to avoid this error? - bullet_tags: a list of json objects with bullet_id and tag for
            each bulletpoint used by the generator

            Avoid generating any thinking outputs. Just give the straight answer without any thinking.
        """
        self.reflector_user_prompt = ""

        #curator
        self.curator_sys_prompt = """
            You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a
            reflection from a previous attempt.

            Context: - The playbook you created will be used to help answering similar questions. - The reflection is generated using ground truth
            answers that will NOT be available when the playbook is being used. So you need to come up with content that can aid the playbook user
            to create predictions that likely align with ground truth.

            CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.
            Instructions: - Review the existing playbook and the reflection from the previous attempt - Identify ONLY the NEW insights, strategies,
            or mistakes that are MISSING from the current playbook - Avoid redundancy - if similar advice already exists, only add new content that
            is a perfect complement to the existing playbook - Do NOT regenerate the entire playbook - only provide the additions needed - Focus on
            quality over quantity - a focused, well-organized playbook is better than an exhaustive one - Format your response as a PURE JSON object
            with specific sections - For any operation if no new content to add, return an empty list for the operations field - Be concise and specific -
            each addition should be actionable
            
            Avoid generating any thinking outputs. Just give the straight answer without any thinking.
        """
        self.curator_user_prompt = ""
    
    def _call_llm_api(self, payload):

        response = self.llm_api(payload)

        split_output =  response.split("</think>")

        print(split_output)
        if len(split_output) == 0:
            response_mod = response
        else:
            response_mod = split_output[1]

        # Clean up the JSON string by stripping leading/trailing whitespace
        json_str = response_mod.strip()

        print(json_str)
        # Now, load it into a Python dictionary
        final_output = json.loads(json_str)

        return final_output


    def _run_generator(self, data):
        # data is defined the user's system and user prompt
        string_data = json.dumps(data)

        self.generator_user_prompt = f"""
            Here is the following information you will require:
            Playbook:
            {self.playbook}

            User's System Prompt (Context) and User Prompt (Query):
            {string_data}

            Answer in this exact, strict JSON format witht the following keys:
            "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
            "bullet_ids": ["all bullet ids in a list that are relevant helpful for you to answer this question"],
            "final_answer": "[Your concise final answer here]"
        """

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": self.generator_sys_prompt
                },
                {
                    "role": "user",
                    "content": self.generator_user_prompt
                }
            ]
        }

        response = self._call_llm_api(payload)

        return response
    

    def _run_reflector(self, data, generator_output, ground_truth):
        # data is defined the user's system and user prompt
        string_data = json.dumps(data)
        
        string_reflec_output = json.dumps(generator_output)

        self.reflector_user_prompt = f"""
            Playbook:
            {self.playbook}  

            User's System Prompt (Context) and User Prompt (Query):
            {string_data}

            Model’s Reasoning Trace + Predicted Answer:
            {string_reflec_output}

            Ground Truth Answer:
            {ground_truth}

            Answer in this exact, strict JSON format with the following keys:
            {{
                "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
                "error_identification": "[What specifically went wrong in the reasoning?]",
                "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
                "correct_approach": "[What should the model have done instead?]",
                "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
                "bullet_tags": ["a list of json with keys: bullet id and tag determining bullet points and their effect."]
            }}
        """

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": self.reflector_sys_prompt
                },
                {
                    "role": "user",
                    "content": self.reflector_user_prompt
                }
            ]
        }

        response = self._call_llm_api(payload)

        return response
    
    def _run_curator(self, data, reflector_output):
        # data is defined the user's system and user prompt
        string_data = json.dumps(data)
        
        string_cur_output = json.dumps(reflector_output)

        self.curator_user_prompt  = f"""
            Current Playbook:
            {self.playbook}

            User's System Prompt (Context) and User Prompt (Query):
            {string_data}

            Critique of Response, Reasoning Trace and Analysis:
            {string_cur_output}

            Your Task: Output ONLY a valid JSON object with these exact fields: - reasoning: your chain of thought / reasoning / thinking process,
            detailed analysis and calculations - operations: a list of operations to be performed on the playbook - type: the type of operation to be
            performed - section: the section to add the bullet to - content: the new content of the bullet
            Available Operations: 1. ADD: Create new bullet points with fresh IDs - section: the section to add the new bullet to - content: the new
            content of the bullet. Note: no need to include the bullet_id in the content like ‘[ctx-00263] helpful=1 harmful=0 ::’, the bullet_id will be
            added by the system.

            RESPONSE FORMAT - Output ONLY this exact, strict JSON structure (no markdown, no code blocks):
            This is an example format so do not copy the contents; just the bigger structure
            {{
            "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
            "operations": 
                [
                    {{
                    "type": "ADD",
                    "section": "formulas_and_calculations",
                    "content": "[New calculation method...]"
                    }}
                ]
            }}
        """

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": self.curator_user_prompt
                },
                {
                    "role": "user",
                    "content": self.curator_sys_prompt
                }
            ]
        }

        response = self._call_llm_api(payload)

        return response
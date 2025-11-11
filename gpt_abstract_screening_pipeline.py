import pandas as pd
import numpy as np
from openpyxl import load_workbook
import sklearn
from openai import OpenAI
from IPython.display import display, HTML
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import main_prompt
import os
import re
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from dotenv import load_dotenv  
load_dotenv()

MAIN_MODEL = "qwen3-235b-a22b"       
CRITIC_MODEL = "deepseek-r1"        
REVISION_MODEL = "mistral-large-instruct"

RELEVANCE_DATASET_COLUMNS = ["short_id", "output"]
relevance_dataset_lock = Lock()
relevance_dataset = pd.DataFrame(columns=RELEVANCE_DATASET_COLUMNS)


def _append_to_relevance_dataset(short_id, output):
    """Store the model's output alongside the article identifier."""
    global relevance_dataset
    new_entry = pd.DataFrame({"short_id": [short_id], "output": [output]})
    with relevance_dataset_lock:
        relevance_dataset = pd.concat([relevance_dataset, new_entry], ignore_index=True)
    return relevance_dataset


def _extract_vote_from_output(output_text):
    if not isinstance(output_text, str):
        return None
    decision_match = re.search(r"Decision:\s*(INCLUDE|EXCLUDE)", output_text, re.IGNORECASE)
    if decision_match:
        return decision_match.group(1).upper()
    return None


def _build_comparison_df(prediction_df, ground_truth_df):
    required_prediction_cols = {"short_id", "output"}
    required_ground_truth_cols = {"short_id", "included"}

    if not required_prediction_cols.issubset(prediction_df.columns):
        missing = required_prediction_cols.difference(prediction_df.columns)
        raise ValueError(f"prediction_df is missing required columns: {missing}")
    if not required_ground_truth_cols.issubset(ground_truth_df.columns):
        missing = required_ground_truth_cols.difference(ground_truth_df.columns)
        raise ValueError(f"ground_truth_df is missing required columns: {missing}")

    merged = pd.merge(
        ground_truth_df[["short_id", "included"]].copy(),
        prediction_df[["short_id", "output"]].copy(),
        on="short_id",
        how="inner",
    )

    merged["vote"] = merged["output"].apply(_extract_vote_from_output)
    if merged["vote"].isnull().any():
        raise ValueError("Could not extract INCLUDE/EXCLUDE decision from one or more model outputs.")

    return merged

def _ensure_placeholders(prompt_text: str) -> str:
    # minimal safety net: append a tiny section for any missing key
    parts = []
    if "{ABSTRACT}" not in prompt_text:
        parts.append("\n\n---\nABSTRACT:\n{ABSTRACT}")
    if "{TITLE}" not in prompt_text:
        parts.append("\n\n---\nTITLE:\n{TITLE}")
    return prompt_text + "".join(parts)


def preprocess_ground_truth_df(file_path): #sheet_name='HVW'sheet_name='HVW'
    df = pd.read_excel(file_path)

    df_train = df.drop(columns=['included', 'article_type'])
    df_test = df[['short_id', 'included']]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df_train, df_test, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#X_train, X_test, y_train, y_test = preprocess_ground_truth_df('abstract_screening.xlsx', sheet_name='HVW')

def initilize_agent():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = "https://chat-ai.academiccloud.de/v1"
    client = OpenAI(api_key = api_key, base_url = base_url)
    return client

def format_prompt(abstract, title, short_id, system_prompt_file_path, user_prompt_file_path):
    file1 = open(system_prompt_file_path, 'r')
    formatted_system_prompt = file1.read()
    file1.close()

    file2 = open(user_prompt_file_path, 'r')
    user_prompt = file2.read()
    file2.close()

    formatted_user_prompt = user_prompt.format(ABSTRACT=abstract, TITLE=title)
    return formatted_system_prompt, formatted_user_prompt, short_id

def critic_format_prompt(title, abstract, chain_of_thought, prediction, ground_truth, previous_system_prompt_file_path, previous_user_prompt_file_path):
    file1 = open(previous_system_prompt_file_path, 'r')
    prev_system_prompt = file1.read()
    file1.close()

    file2 = open(previous_user_prompt_file_path, 'r')
    prev_user_prompt = file2.read()
    file2.close()

    formatted_user_prompt = critic_prompt.CRITIC_USER.format(
        TITLE=title, ABSTRACT=abstract, CHAIN_OF_THOUGHT=chain_of_thought, 
        VOTE=prediction, DECISION=ground_truth, SYSTEM_PROMPT=prev_system_prompt, USER_PROMPT = prev_user_prompt)
    
    return formatted_user_prompt

def revision_format_prompt(title, abstract, agent_reasoning, wrong_decision, correct_label, critic_feedback, previous_system_prompt_file_path, previous_user_prompt_file_path):
    file1 = open(previous_system_prompt_file_path, 'r')
    prev_system_prompt = file1.read()
    file1.close()

    file2 = open(previous_user_prompt_file_path, 'r')
    prev_user_prompt = file2.read()
    file2.close()

    formatted_user_prompt = revision_prompt.REVISION_USER_PROMPT.format(
        title=title, abstract=abstract, agent_reasoning=agent_reasoning,
        wrong_decision=wrong_decision, correct_label=correct_label,
        critic_feedback=critic_feedback, original_system_prompt=prev_system_prompt,
        original_user_prompt=prev_user_prompt)
    
    return formatted_user_prompt

def get_relevance_score(MODEL, formatted_system_prompt, formatted_user_prompt, short_id):
    client = initilize_agent()
    chat_completion = client.chat.completions.create(messages=[{"role":"system","content": formatted_system_prompt},
                                                            {"role":"user","content": formatted_user_prompt}], model='openai-gpt-oss-120b')#'qwen3-235b-a22b')

    output = chat_completion.choices[0].message.content.strip()
    # so that the API rate limit is not exceeded
    time.sleep(8)
    print(f"Processed {short_id} with output: {output}")
    updated_dataset = _append_to_relevance_dataset(short_id, output)
    get_relevance_score.dataset = updated_dataset
    return updated_dataset

def evaluation(prediction_df, ground_truth_df):
    comparison_df = None
    comparison_df = _build_comparison_df(prediction_df, ground_truth_df)
    print("Comparison DataFrame:")
    print(comparison_df)
    comparison_df['y_true'] = comparison_df['included'].map({'yes': 1, 'no': 0})
    comparison_df['y_pred'] = comparison_df['vote'].map({'INCLUDE': 1, 'EXCLUDE': 0})

    cm = confusion_matrix(comparison_df['y_true'], comparison_df['y_pred'])
    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(comparison_df['y_true'], comparison_df['y_pred'])
    precision = precision_score(comparison_df['y_true'], comparison_df['y_pred'])
    recall = recall_score(comparison_df['y_true'], comparison_df['y_pred'])

    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=['Actual no','Actual yes'], columns=['Pred EXCLUDE','Pred INCLUDE']))
    print()
    print(f"True Positive (TP): {TP}")
    print(f"True Negative (TN): {TN}")
    print(f"False Positive (FP): {FP}")
    print(f"False Negative (FN): {FN}")
    print()
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['EXCLUDE', 'INCLUDE'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix — INCLUDE vs EXCLUDE')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

    false_clf_df = comparison_df[comparison_df['y_true'] != comparison_df['y_pred']]
    return false_clf_df


import critic_prompt

def critic_agent(false_clf_df, previous_system_prompt_file_path, previous_user_prompt_file_path):
    client = initilize_agent()
    print("______________________________")
    print(false_clf_df)
    print("______________________________")
    short_id = false_clf_df['short_id']
    ground_truth_df = pd.read_excel("V1_abstract_screening.xlsx") ## include this as a general argument to the function
    paper_title = ground_truth_df.loc[ground_truth_df.short_id == short_id].title
    paper_abstract = ground_truth_df.loc[ground_truth_df.short_id == short_id].abstract

    formatted_critic_user_prompt = critic_format_prompt(
        title=paper_title, #false_clf_df['title'],
        abstract=paper_abstract, #false_clf_df['abstract'],
        chain_of_thought=false_clf_df["output"],
        prediction=false_clf_df["vote"],
        ground_truth=false_clf_df["included"],
        previous_system_prompt_file_path=previous_system_prompt_file_path,
        previous_user_prompt_file_path=previous_user_prompt_file_path)
    
    chat_completion = client.chat.completions.create(messages=[{"role":"system","content": critic_prompt.CRITIC_SYSTEM},
                                                           {"role":"user","content": formatted_critic_user_prompt}],
                                                           model='openai-gpt-oss-120b')#'deepseek-r1')

    critic_output = chat_completion.choices[0].message.content.strip()
    time.sleep(1)
    return critic_output

import revision_prompt

def revision_agent(false_clf_df, feedback, previous_system_prompt_file_path, previous_user_prompt_file_path):
    client = initilize_agent()

    short_id = false_clf_df['short_id']
    ground_truth_df = pd.read_excel("V1_abstract_screening.xlsx") ## include this as a general argument to the function
    paper_title = ground_truth_df.loc[ground_truth_df.short_id == short_id].title
    paper_abstract = ground_truth_df.loc[ground_truth_df.short_id == short_id].abstract

    formatted_user_prompt = revision_format_prompt(
        title=paper_title, #false_clf_df['title'],
        abstract=paper_abstract, #false_clf_df['abstract'],
        agent_reasoning=false_clf_df["output"],
        wrong_decision=false_clf_df["vote"],
        correct_label=false_clf_df["included"],
        critic_feedback=feedback,
        previous_system_prompt_file_path=previous_system_prompt_file_path,
        previous_user_prompt_file_path=previous_user_prompt_file_path)
    
    system_prompt = revision_prompt.REVISION_SYSTEM_PROMPT

    chat_completion = client.chat.completions.create(messages=[{"role":"system","content": system_prompt},
                                                           {"role":"user","content": formatted_user_prompt}],
                                                           model='openai-gpt-oss-120b')#'mistral-large-instruct')

    revised_prompt = chat_completion.choices[0].message.content.strip()
    print("Revised Prompt Generated.")
    print(revised_prompt)
    time.sleep(1)

    return revised_prompt

def save_prompt(revised_prompt, old_file_path):
    name = old_file_path.split('.')[1]
    prompt_version = int(name[-1])
    prompt_version += 1
    if "system" in old_file_path:
        filename = "system_prompt_v" + str(prompt_version) + ".txt"
        with open("./prompt_log/system_prompt_v" + str(prompt_version) + '.txt', 'w') as file:
            file.write(revised_prompt)
    else:
        formatted_revised_prompt = _ensure_placeholders(revised_prompt)
        filename = "user_prompt_v" + str(prompt_version) + ".txt"
        with open("./prompt_log/user_prompt_v" + str(prompt_version) + '.txt', 'w') as file:
            file.write(formatted_revised_prompt)
    return filename, prompt_version

    # Example usage:
    # df['Relevance Score'] = df['Abstract'].apply(get_relevance_score)


    #get_relevance_score.dataset = relevance_dataset
# fix the issue that in the next iteration the false clf is already created
VIEWED_SHORT_IDS = set()
def full_iteration(ground_truth_df, file_path_sys_prompt, file_path_user_prompt, false_clf_df=None):
    X_train, X_test, y_train, y_test = preprocess_ground_truth_df(file_path=ground_truth_df) #  sheet_name='HVW' (file_path='abstract_screening.xlsx', sheet_name='HVW')
    # For demonstration, we will use only the first 200 entries from the training set
    X_train_llm = X_train[:50] #200
    y_train_llm = y_train[:50] #200

    abstracts = X_train_llm['abstract'].tolist()
    titles = X_train_llm['title'].tolist()
    short_ids = X_train_llm['short_id'].tolist()

    for i in range(0, 3): #range(len(short_ids)) # for testing purpose, we will use only the first 3 entries
        print(f"Entry {i}: short_id {short_ids[i]}")
        formatted_system_prompt, formatted_user_prompt, short_id = format_prompt(abstracts[i], titles[i], short_ids[i], file_path_sys_prompt, file_path_user_prompt) # "./prompt_log/system_prompt_v1.txt", "./prompt_log/user_prompt_v1.txt"
        model = 'openai-gpt-oss-120b'
        updated_dataset = None
        updated_dataset = get_relevance_score(MODEL=model, formatted_system_prompt=formatted_system_prompt, formatted_user_prompt=formatted_user_prompt, short_id=short_id)
    print(updated_dataset)

    #incase you want to save the dataset
    #updated_dataset.to_csv("relevance_dataset_examples.csv", index=False)
    #updated_dataset = pd.read_csv("relevance_dataset_examples.csv")

    print("Evaluation on training set:")
    false_clf_df = None
    false_clf_df = evaluation(updated_dataset, y_train_llm)
    print("Critic agent analysis on false classifications:")

    filtered_false_clf_df = false_clf_df[~false_clf_df['short_id'].isin(VIEWED_SHORT_IDS)]
    if len(filtered_false_clf_df) == 0:
        print("No new false classifications to analyze.")
        return false_clf_df, file_path_sys_prompt, file_path_user_prompt
    else:
        row = filtered_false_clf_df.iloc[0]
        # ----- FIX THIS -----
        print(f"Analyzing false classification for short_id: {row['short_id']}")
        critic_output = critic_agent(row, previous_system_prompt_file_path="./prompt_log/system_prompt_v1.txt", previous_user_prompt_file_path="./prompt_log/user_prompt_v1.txt")
        print(f"Critic Output:\n{critic_output}\n")
        print("revising the prompt...")
        revised_output = revision_agent(row, feedback=critic_output, previous_system_prompt_file_path="./prompt_log/system_prompt_v1.txt", previous_user_prompt_file_path="./prompt_log/user_prompt_v1.txt")
        print(f"Revised Output:\n{revised_output}\n")

        VIEWED_SHORT_IDS.add(row['short_id'])

        revised_prompts = revised_output.split("---REVISED USER PROMPT---")
        #prompts = revised_output.split("PROMPT---")
        revised_user_prompt = revised_prompts[1]
        #revised_system_prompt = prompts[0].split("---REVISED SYSTEM PROMPT---")[1]
        revised_system_prompt = revised_prompts[0]#.split("---REVISED SYSTEM PROMPT---")[1]

        user_prompt_filename, prompt_version = save_prompt(revised_user_prompt, file_path_user_prompt)
        system_prompt_filename, prompt_version = save_prompt(revised_system_prompt, file_path_sys_prompt)
        print("revised user prompt: " + revised_user_prompt)
        print("revised system prompt: " + revised_system_prompt)
        print("saved under: " + user_prompt_filename + " and " + system_prompt_filename + " version: " + str(prompt_version))
        time.sleep(1)

        print("Process completed. Horray! ")
    return false_clf_df, system_prompt_filename, user_prompt_filename


def main():
    # denk nochmal über die logik von dieser funktion nach -> wie kann man garantieren, dass die iteration die korrekten agrumente bekommt
    ground_truth_df = "V1_abstract_screening.xlsx"
    system_prompt_filename = "./prompt_log/system_prompt_v1.txt"
    user_prompt_filename = "./prompt_log/user_prompt_v1.txt"
    false_clf_df = None
    for i in range(3):
        false_clf_df = None
        false_clf_df, system_prompt_filename, user_prompt_filename = full_iteration(ground_truth_df, system_prompt_filename, user_prompt_filename, false_clf_df)
        print("file saved as: " + system_prompt_filename + " and " + user_prompt_filename)

        print(f"Completed iteration {i+1}\n\n")
        system_prompt_filename = "./prompt_log/" + system_prompt_filename
        user_prompt_filename = "./prompt_log/" + user_prompt_filename
        print("system prompt for next iteration: " + system_prompt_filename)
        print("user prompt for next iteration: " + user_prompt_filename)
        time.sleep(3)
        if false_clf_df is None or false_clf_df.empty:
            print("No false classifications to process. Ending iterations.")
            break


main()

# problem: in the followinf iterations the abtracts arent correctly passed to the formatted prompt
# refer to the chatgpt history for more details
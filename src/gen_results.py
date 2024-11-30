from openai import OpenAI
from datasets import load_dataset

client_OpenAI = OpenAI(
    api_key="",
)

def get_eval_gpt4o(prompt, system_prompt, model="gpt-4o"):
    response = client_OpenAI.chat.completions.create(
        model=model,
        messages=[{
                    'role': 'system',
                    'content': system_prompt
                },{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()

def run_batches(dataset):

    # dataset = data.copy()

    dataset = dataset.reset_index(drop=True)

    system_prompt="""Be concise and answer in less than 15 words:"""

    dataset['body'] = [{"model":"gpt-4-turbo", "max_tokens": 100,
        "messages": [{'role': 'system','content': system_prompt},
                    {'role': 'user', 'content':f"{dataset['Question'][i]}" }] } for i in range(len(dataset))]

    dataset['method'] = 'POST'

    dataset["url"]= "/v1/chat/completions"

    openai_dataset = dataset[['body', 'custom_id', 'method', 'url']]

    openai_dataset.to_json('eval_data.jsonl', orient='records', lines=True)

    batch_input_file = client_OpenAI.files.create(
      file=open("/content/eval_data.jsonl", "rb"),
      purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch_id = client_OpenAI.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
          "description": "nightly eval job"
        }
    )

    return batch_id.id

if __name__ == '__main__':
    # Load the dataset
    ds = load_dataset("gagan3012/DateLogicQA")

    dataset = ds['train'].to_pandas()

    dataset['custom_id'] = ["Request-"+str(i) for i in range(len(dataset))]

    batch_ids = []

    deldataset = dataset.copy()
    import time

    # # batch_ids = batch_ids['batch_id'].tolist()

    while len(deldataset) > 0:
        # Take the first 100 items or all remaining items if less than 100
        batch = deldataset.head(100)

        # Run the batch and store the batch ID
        batch_id = run_batches(batch)
        batch_ids.append(batch_id)

        # Remove the processed items from the dataset
        deldataset = deldataset.iloc[100:]

        # Optional: Print progress
        print(f"Processed batch of {len(batch)} items. {len(deldataset)} items remaining.")

        time.sleep(90)
import torch
from transformers import pipeline, AutoTokenizer

model_path = "/models/tsuzumi-7b-v1_2-8k-instruct"

SYSTEM_PROMPT = "あなたは有益なAIアシスタントです。質問や指示を良く読んで従ってください。様々な観点からできる限り詳細に説明してください。\n"
text = "四国地方の４つの都道府県名と、それぞれの県庁所在地を列挙してください。"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
generation_params = {
    'do_sample': True,
    'max_new_tokens': 512,
    'temperature': 0.3,
    'top_p': 0.9,
    'use_cache': True,
    'return_full_text': False,
    'pad_token_id': tokenizer.pad_token_id,
    'eos_token_id': tokenizer.eos_token_id
}

generator = pipeline("text-generation", \
                     model_path, \
                     tokenizer=tokenizer, \
                     torch_dtype=torch.bfloat16, \
                     device_map="auto", \
                     trust_remote_code=True
                    )

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": text},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
)

output = generator(prompt, **generation_params)

messages = output[0]["generated_text"]
print(messages)

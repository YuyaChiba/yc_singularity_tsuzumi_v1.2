import torch
from transformers import pipeline, AutoTokenizer

model_path = "/models/tsuzumi-7b-v1_2-8k-instruct"

SYSTEM_PROMPT = "あなたは有益なAIアシスタントです。質問や指示を良く読んで従ってください。\n"
text = """
次の文章を読んで、英語のYes/No問題を作ってください。

 I was fed up with doing things I didn't want to do. Homework, tidying up, helping around the house - how boring! I'd much rather have fun instead. One day, I had a great idea. I would build another me. Then he could do everything for me! So I spent all my pocket money and bought a robot. 'I'd like to buy your cheapest robot, please!' Sale! Sale Sale Sale Sale! On the way home, I told the robot about my plan. 'From now on, you're going to be the new me!' 'Yes, sir!' 'But don't let anyone know. You must behave exactly like me.' 'I can do that! But first I need to know everything about you.' 'Hmm, that's a bit tricky. I don't know where to start.' 'Perhaps we should begin by making a list of facts about you...'
"""

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

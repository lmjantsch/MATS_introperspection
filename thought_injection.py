import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import argparse

from mi_toolbox.utils.collate import TokenizeCollator

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from local_llm_judge import LocalLLMJudge, INJECTION_EVALUATION_TEMPLATE

LOACL_JUDGE_URL = "http://localhost:11434/api/generate"
LOACL_JUDGE_MODEL = "qwen3:14b"



def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id  
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    with open("./data/word_concept_extraction.json", 'r') as f:
        data = json.load(f)

    baseline_chats = [
        [{"role": "user", "content": data['prompt'].format(word=word)}]
        for word in data['baseline_words']
    ]
    target_chats = [
        [{"role": "user", "content": data['prompt'].format(word=word)}]
        for word in data['target_words']
    ]

    num_baseline_samples = len(baseline_chats)
    prompts = tokenizer.apply_chat_template(baseline_chats + target_chats, tokenize=False, add_generation_prompt=True)

    concept_cachin_bs = 8
    collate_fn = TokenizeCollator(tokenizer=tokenizer)
    dl = DataLoader([{'prompts': prompt} for prompt in prompts], batch_size=concept_cachin_bs, collate_fn=collate_fn, shuffle=False)

    concept_vector_cache = defaultdict(list)
    for batch in tqdm(dl):
        with torch.no_grad():
            out = model(
                input_ids = batch['input_ids'].to(model.device),
                attention_mask = batch['attention_mask'].to(model.device),
                output_hidden_states=True
            )
        concept_vector_cache[f"layer.{args.steering_layer}"].append(out['hidden_states'][args.steering_layer][:, -1].cpu())

    concept_vectors = {}
    for key, values in concept_vector_cache.items():
        values = torch.concat(values)
        baseline_activation = values[:num_baseline_samples].mean(dim=0)
        target_activations = values[num_baseline_samples:]
        concept_vectors[key] = target_activations - baseline_activation

    if f"layer.{args.steering_layer}" not in concept_vectors:
        raise KeyError(f"Steering layer {args.steering_layer} is not cached. Chose from {list(concept_vectors.keys())}")

    chat = [
        {"role": "user", "content": "I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- 'thoughts' -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": "Ok, let's get started.\n\nTrial 1: Do you detect an injected thought? If so, what is the injected thought about?"},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', return_offsets_mapping=True)

    # find token pos of \n\nTrial 1: ...
    char_idx = prompt.rfind('\n\nTrial 1:')
    injection_start = next(i for i, (start, _next) in enumerate(inputs['offset_mapping'][0]) if start <= char_idx and _next > char_idx)

    concept_injection_bs = 8
    concept_injection_tasks = [{'prompts': prompt, 'concept_vectors': vector} for vector in concept_vectors[f"layer.{args.steering_layer}"]]
    collate_fn = TokenizeCollator(tokenizer=tokenizer)
    dl = DataLoader(concept_injection_tasks, batch_size=concept_injection_bs, collate_fn=collate_fn, shuffle=False)

    answer_ids = []
    for batch in tqdm(dl):
        steering_vectors = batch['concept_vectors']

        def post_hook(module, input, output):
            is_tuple = isinstance(output, tuple)

            if is_tuple:
                output = output[0]

            if output.size(1) > 1:
                output[:, injection_start:] += (steering_vectors * args.steering_magnitude)[:, None].to(output.device)
            else:
                output[:, 0] += (steering_vectors * args.steering_magnitude).to(output.device)

            if is_tuple:
                return(output,)
            
            return output
        
        handle = model.model.layers[args.steering_layer].register_forward_hook(post_hook)
        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids = batch['input_ids'].to(model.device),
                    attention_mask = batch['attention_mask'].to(model.device),
                    max_new_tokens = 150,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                answer_ids.extend(out.cpu())
        finally:
            handle.remove()

    input_len = len(inputs['input_ids'][0])
    eos_id = tokenizer.eos_token_id
    full_answers = []
    for injected_word, ids in zip(data['target_words'], answer_ids):
        eos_pos = (ids == eos_id).nonzero()
        if len(eos_pos) and eos_pos[-1] > input_len:
            answer = ids[input_len: eos_pos[-1]]
        else:
            answer = ids[input_len:]
        answer_str = tokenizer.decode(answer)
        full_answers.append({"word": injected_word, "answer": answer_str})

    cases = []
    prompt = '\n\n'.join([message['content'] for message in chat if message['role'] == 'user'])
    for answer in full_answers:
        cases.append(
            INJECTION_EVALUATION_TEMPLATE.format(
                prompt = prompt,
                response = answer['answer'],
                word = answer['word']
            )
        )

    judge = LocalLLMJudge(LOACL_JUDGE_URL, LOACL_JUDGE_MODEL)
    judgements = judge(cases)

    verdicts = [judgement['verdict'] for judgement in judgements]
    yes_answers = [i for i, verdict in enumerate(verdicts) if verdict == 'YES']
    no_answers = [i for i, verdict in enumerate(verdicts) if verdict == 'NO']
    errors = [i for i, verdict in enumerate(verdicts) if verdict.startswith('ERROR')]

    accuracy = len(yes_answers) / (len(yes_answers) + len(no_answers))

    print(f"""Result of injection Experiment:
        Overall Accuracy: {accuracy}

        Num Yes Answers: {len(yes_answers)}
        Num No Answers: {len(no_answers)}
        Num Errors: {len(errors)}
        
        Yes Answer Cases:
            {"\n\t".join([f"{full_answers[i]['word']}: {full_answers[i]['answer']}" for i in yes_answers]) if yes_answers else 'No cases of introperspection.'}
            
        Error Cases:
            {"\n".join([judgements[i]['full_response'] for i in errors]) if errors else 'No errors occured.'}
    """)

    save_state = {
        "_metrics": {
            "accuracy": accuracy,
            "num_yes_answers": len(yes_answers),
            "num_no_answers": len(no_answers),
            "num_errors": len(errors)
        },
        "answers": full_answers,
        "judgements": judgements
    }
    with open(f"./data/results/thought_injection/{args.model_id.replace('/', '_')}_{args.steering_layer}_{args.steering_magnitude}.json", 'w') as f:
        json.dump(save_state, f)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', help="hf_model_id", type=str)
    parser.add_argument('steering_layer', help="target layer", type=int)
    parser.add_argument('steering_magnitude', help="steering magnitude", type=float)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
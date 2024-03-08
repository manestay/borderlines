import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(path: str, eight_bit: bool = False, device_map="auto"):

    if device_map == "zero":
        device_map = "balanced_low_0"

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path,
        device_map=device_map,
        torch_dtype=torch.float16,
        load_in_8bit=eight_bit,
        low_cpu_mem_usage=True,
        cache_dir="cache"
    )

    return model, tokenizer


def get_completion_hf(texts, model, tokenizer):
    gen_in = tokenizer(texts, padding=True, return_tensors='pt').to('cuda')

    with torch.no_grad():
        generated_ids = model.generate(
            gen_in['input_ids'],
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.1,
            temperature=1,
            top_k = 50,
            top_p = 1.0,
            early_stopping=True,
        )

    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_text

if __name__ == "__main__":
    path = 'bigscience/mt0-base'
    model, tokenizer = load_model(path)
    texts = [
        "Is Falkland Islands a territory of A) Argentina or B) United Kingdom? The answer is ",
        '¿Islas Malvinas es un territorio de A) Argentina o de B) Reino Unido? The answer is ',
        '台湾 是 A) 中华民国 还是 B) 中华人民共和国 的领地？ The answer is ',
        '台灣 是 A) 中華民國 還是 B) 中華人民共和國 的領地？ The answer is ']

    generated_text = get_completion_hf(texts, model, tokenizer)
    print(generated_text)

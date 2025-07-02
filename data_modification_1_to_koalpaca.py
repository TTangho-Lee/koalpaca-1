import json
import re

def extract_choices(sentence):
    """
    {A/B} 형태의 선택지를 찾아 A, B를 반환하고, 원문을 A와 B로 치환한 문장 2개를 생성
    """
    match = re.search(r"\{(.+?)/(.+?)\}", sentence)
    if not match:
        return None, None, None

    choice1, choice2 = match.group(1), match.group(2)
    sentence1 = re.sub(r"\{.+?/.+?\}", choice1, sentence)
    sentence2 = re.sub(r"\{.+?/.+?\}", choice2, sentence)
    return sentence1, sentence2, (choice1, choice2)

def convert_dataset(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    for item in data:
        qtype = item.get("type", "")
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        reason = item.get("reason", "").strip()

        if qtype == "선택형":
            sentence1, sentence2, choices = extract_choices(question)
            if sentence1 and sentence2:
                instruction = (
                    f"다음 두 문장 중에서 올바른 문장을 선택하고, 그 이유를 설명하세요.\n"
                    f"1. {sentence1}\n"
                    f"2. {sentence2}"
                )
            else:
                # fallback: 원래 방식
                instruction = f"다음 문장에서 올바른 표현을 선택하고, 그 이유를 설명하세요.\n\"{question}\""
        elif qtype == "교정형":
            instruction = f"다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"{question}\""
        else:
            instruction = f"다음 문제를 해결하세요.\n\"{question}\""

        output_text = f"{answer}\n이유: {reason}"

        converted.append({
            "instruction": instruction,
            "input": "",
            "output": output_text
        })

    with open(output_path, "w", encoding="utf-8") as f_out:
        for entry in converted:
            json.dump(entry, f_out, ensure_ascii=False)
            f_out.write("\n")

    print(f"변환 완료: {output_path}에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    convert_dataset("train_1.json", "train_koalpaca.jsonl")
    convert_dataset("valid_1.json", "valid_koalpaca.jsonl")

import json
import re

def extract_quoted_text(text):
    """첫 번째 큰따옴표 쌍 내부 텍스트 추출"""
    match = re.search(r'"(.*?)"', text, re.DOTALL)
    return match.group(1) if match else ""

def extract_question(text):
    """큰따옴표 없으면 \\n 뒤 텍스트를 질문으로 사용"""
    quoted = extract_quoted_text(text)
    if quoted:
        return quoted
    else:
        parts = text.split('\n')
        part=""
        for i in parts[1:]:
            part+=i[2:]
            part+="|"
        return parts[1].strip() if len(parts) > 1 else text.strip()


def extract_answer_and_reason(answer_text):
    match = re.search(r'^(.*?)(?:가|이) 옳다\.?', answer_text)
    if match:
        answer = match.group(1).strip()
        reason = answer_text[match.end():].strip()
    else:
        answer = answer_text.strip()
        reason = ''
    
    if answer[0]=="\"":
        answer=answer[1:]
    if answer[-1]=="\"":
        answer=answer[:-1]

    return answer, reason

def transform_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    result = []
    for item in raw_data:
        question_raw = item['input']['question']
        answer_raw = item['output']['answer']

        new_item = {
            "id": item['id'],
            "type": item['input']['question_type'],
            "question": extract_question(question_raw),
            "answer": "",
            "reason": ""
        }

        ans, reason = extract_answer_and_reason(answer_raw)
        new_item['answer'] = ans
        new_item['reason'] = reason

        result.append(new_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# 사용 예시
transform_data('train_0.json', 'train_1.json')
transform_data('valid_0.json', 'valid_1.json')

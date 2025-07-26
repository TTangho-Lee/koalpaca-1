'''
주석처리하기
제작한 db 방식
{
  "id": "punctuation_bracket",
  "title": "문장 부호 - 대괄호([ ]) 규정",
  "content": [
    "(1) 괄호 안에 또 괄호를 쓸 필요가 있을 때 바깥쪽의 괄호로 쓴다.",
    "(2) 고유어에 대응하는 한자어, 외래어, 외국어 표기임을 나타낼 때 쓴다.",
    "(3) 원문에 대한 이해를 돕기 위해 설명이나 논평 등을 덧붙일 때 쓴다."
  ],
  "examples": [
    "어린이날이 새로 제정되었을 당시에는 어린이들에게 경어를 쓰라고 하였다.[윤석중 전집(1988), 70쪽 참조]",
    "이번 회의에는 두 명[이혜정(실장), 박철용(과장)]만 빼고 모두 참석했습니다.",
    "나이[年歲]", "낱말[單語]", "손발[手足]", "낱말[word]", "문장[sentence]", "책[book]", "독일[도이칠란트]",
    "국제 연합[유엔]", "자유 무역 협정[FTA]", "에프티에이(FTA)", "국제 연합 교육 과학 문화 기구[UNESCO]",
    "유네스코(UNESCO)", "국제 연합[United Nations]", "유엔(United Nations)",
    "그것[한글]은 이처럼 정보화 시대에 알맞은 과학적인 문자이다.",
    "신경준의 ≪여암전서≫에 \"삼각산은 산이 모두 돌 봉우리인데, 그 으뜸 봉우리를 구름 위에 솟아 있다고 백운(白雲)이라 하며 [이하 생략]\"",
    "그런 일은 결코 있을 수 없다.[원문에는 ‘업다’임.]"
  ],
  "exceptions": [],
  "tags": {
    "morpheme": [
      "기호([, ])",
      "명사(NNG, NNP)",
      "한자어, 외래어, 외국어(NNG, SL, SH, SF 등)",
      "주석/설명어구",
      "특수목적어(설명, 논평, 해설 등)"
    ],
    "pattern": [
      "대괄호 사용",
      "붙여쓰기(여는 대괄호는 뒷말에, 닫는 대괄호는 앞말에)",
      "주석 추가",
      "다중 괄호(중첩)",
      "고유어-한자어/외국어 병렬 표기"
    ],
    "meta": [
      "문장 부호",
      "주석, 해설, 대응어 병기",
      "띄어쓰기 규정",
      "중첩·다국어 표기"
    ],
    "fallback": true,
    "situation": [
      "주석/부연설명, 대응어 병기 등 특수한 표기가 필요할 때 대괄호 사용",
      "형태소 분석 결과 괄호 내 다국어, 한자어, 약어, 논평 등 특수 설명어 검출 시",
      "대괄호 중첩 필요시 내부에 괄호 또 쓸 수 있음(구조 분석 기반)",
      "붙여쓰기 규정 위반(띄어쓰기 등)이나 설명/병기 목적 외 사용시 예외(fallback)"
    ]
  }
}

코드과정
1. input 에 관련한 db 조항 찾기
2. 찾은 조항 + 프롬프트 + input 을 solar 모델에 넣기

'''


import json
from typing import Dict, Any, List, Optional
from konlpy.tag import Okt  # 형태소 분석기 예시 (Konlpy)
# 파일 최상단
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

checkpoint_path = "./solar_eos/checkpoint-6531"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 8bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,      # 4bit는 load_in_4bit=True
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True  # 오프로드 옵션(필수는 아님)
)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    device_map="auto",
    quantization_config=bnb_config,   # 여기서 설정 전달
    low_cpu_mem_usage=True,
    trust_remote_code=True            # 필요에 따라 추가
)
text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
    batch_size=1
)



# 1. db를 파일에서 로드하는 함수
def load_db(filepath: str) -> List[Dict[str, Any]]:
    print(f"[LOG] 데이터베이스({filepath}) 로드 중...")
    with open(filepath, "r", encoding="utf-8") as f:
        result = json.load(f)
    print(f"[LOG] 데이터베이스 로드 완료. {len(result)}개 규범.")
    return result

# 2. 형태소 분석 및 품사 태그 추출 (Okt 기준)
def morphological_analysis_with_pos(text: str) -> List[Dict[str, str]]:
    print("[LOG] 형태소 분석 시작...")
    okt = Okt()
    morphs_pos = okt.pos(text, norm=True, stem=True)
    result = [{"morph": m, "pos": p} for m, p in morphs_pos]
    print(f"[LOG] 형태소 분석 완료: {result}")
    return result

# 3. 규범 DB 내 morpheme 태그와 유사도 비교 함수
def calculate_similarity(tokens_with_pos: List[Dict[str, str]], morpheme_tags: List[str]) -> float:
    token_terms = set()
    for item in tokens_with_pos:
        token_terms.add(item["morph"].lower())
        token_terms.add(item["pos"].lower())

    morpheme_terms = set(tag.lower() for tag in morpheme_tags)
    matched = token_terms.intersection(morpheme_terms)
    score = float(len(matched))

    return score

# 4. pattern, meta, situation 태그 활용 함수
def calculate_additional_score(tokens_with_pos: List[Dict[str, str]], norm: Dict[str, Any]) -> float:
    text_all_tags = " ".join(
        sum(
            [norm["tags"].get("pattern", []), norm["tags"].get("meta", []), norm["tags"].get("situation", [])],
            []
        )
    ).lower()

    count = 0
    for token_dict in tokens_with_pos:
        morph = token_dict["morph"].lower()
        if morph in text_all_tags:
            count += 1
    total_tokens = max(len(tokens_with_pos), 1)
    score = count / total_tokens

    return score

# 5. 규범 검색: morpheme 유사성과 추가 점수 기반으로 우선순위 정렬
def find_best_norms(tokens_with_pos: List[Dict[str, str]], db: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("[LOG] 규범 유사도 평가 시작...")
    scored_norms = []
    for norm in db:
        morpheme_tags = norm.get("tags", {}).get("morpheme", [])
        fallback = norm.get("tags", {}).get("fallback", False)

        morpheme_score = calculate_similarity(tokens_with_pos, morpheme_tags)
        additional_score = calculate_additional_score(tokens_with_pos, norm)
        score = morpheme_score * 0.7 + additional_score * 0.3
        if fallback:
            score *= 0.5
        scored_norms.append((score, norm))

    scored_norms.sort(key=lambda x: x[0], reverse=True)
    best_norms = [item[1] for item in scored_norms if item[0] > 0.0][:3]
    print(f"[LOG] 선정된 규범 {len(best_norms)}개")
    return best_norms

# 6. 출력 생성 (입력 문장과 가장 관련 있는 규범 여러 개를 활용)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def generate_with_solar(input_data: Dict[str, Any], best_norms: List[Dict[str, Any]]) -> str:
    import gc
    import torch

    torch.cuda.empty_cache()
    gc.collect()
    print(f"[LOG] LLM 프롬프트 생성 및 모델 호출 준비...")
    norms_texts = []
    for norm in best_norms:
        examples = norm.get('examples', {})
        example_sentences = []
        if isinstance(examples, dict):
            for val in examples.values():
                if isinstance(val, list):
                    example_sentences.extend(val)
                elif isinstance(val, str):
                    example_sentences.append(val)
        elif isinstance(examples, list):
            example_sentences.extend(examples)
        example_text = ", ".join(example_sentences)
        norm_text = (
            f"규범 내용: {norm.get('content', '')}\n"
            f"설명: {norm.get('note', '')}\n"
            f"예시: {example_text}"
        )
        norms_texts.append(norm_text)


    if input_data.get("question_type") == "교정형":
    # 교정형일 때만 교정 지시문 포함
        instruct = "문장에서 어문 규범에 맞지 않는 부분을 찾아 고치고, 그 이유를 설명하세요."
        prompt = (
            f"다음은 표준어 규범 정보입니다.\n{chr(10).join(norms_texts)}\n\n"
            f"아래 질문에 답하십시오.\n{instruct}\n질문: {input_data.get('question', '')}\n답변:"
        )
    else:
        # 그 외(선택형 포함)는 기존 prompt 유지
        prompt = (
            f"다음은 표준어 규범 정보입니다.\n{chr(10).join(norms_texts)}\n\n"
            f"아래 질문에 답하십시오.\n질문: {input_data.get('question', '')}\n답변:"
        )


    # 이미 선언된 글로벌 text_gen 사용
    output = text_gen(prompt)[0]['generated_text']

    generated_text = output[len(prompt):].strip()
    # <|endoftext|> 토큰 위치 찾기
    end_token = "<|endoftext|>"
    if end_token in generated_text:
        generated_text = generated_text.split(end_token)[0].strip()

    print("[LOG] 모델로부터 답변 생성 완료.")
    return generated_text



# 7. 메인 함수: 파일 db경로 받음, input 데이터 받아서 답변 반환
def answer_question_with_solar(input_data: Dict[str, Any], db_filepath: str) -> Dict[str, Any]:
    print("[LOG] 질의 답변 프로세스 시작")
    db = load_db(db_filepath)
    tokens_with_pos = morphological_analysis_with_pos(input_data.get("question", ""))
    best_norms = find_best_norms(tokens_with_pos, db)
    checkpoint = "./solar_eos/checkpoint-6531"
    answer_text = generate_with_solar(input_data, best_norms)
    print("[LOG] 모든 프로세스 완료, 결과 반환")
    return {"answer": answer_text}

# --- 사용 예시 (실제 환경에서는 아래 코드 대신 API 핸들러 등에 적용) ---
if __name__ == "__main__":
    import pprint

    db_path = "korean_standard_language_rules.json"  # DB JSON 경로 (실제 경로로 교체)

    # 입력 예시
    input_data_select = {
        "question_type": "선택형",
        "question": "\"나는 그를 본 적이 있음을 {기억해냈다/기억해 냈다}.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요."
    }

    input_data_correct = {
        "question_type": "교정형",
        "question": "다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"오늘은 퍼즐 마추기를 해 볼 거예요.\""
    }

    print("=== 선택형 예시 답변 ===")
    pprint.pprint(answer_question_with_solar(input_data_select, db_path))

    print("\n=== 교정형 예시 답변 ===")
    pprint.pprint(answer_question_with_solar(input_data_correct, db_path))

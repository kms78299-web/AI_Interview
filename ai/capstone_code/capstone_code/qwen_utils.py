"""
Qwen2.5-Omni-7B 로딩 + 채팅 호출 + JSON 파싱.

/ceph_data/kms78299/repos/Various_MLLM/Qwen2.5-Omni-7B/code/model_utils.py 의
패턴을 그대로 따른다 (Thinker만 사용, run_chat/parse_json_object 시그니처 동일).
contents에 {"type": "video", "path": ...}를 넣으면 Qwen2_5OmniProcessor가 프레임
샘플링 + (use_audio_in_video=True일 때) 영상에 포함된 오디오까지 같이 처리한다 —
숫자 통계만 텍스트로 설명해주는 게 아니라 모델이 실제 화면/음성을 직접 보고 판단하게
하기 위함.
"""

import json

import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration


def load_qwen_omni_thinker(model_id):
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    return model, processor


def _eos_token_ids(processor):
    """기본 eos 말고 Qwen 채팅 템플릿의 turn-end 토큰(<|im_end|> 등)도 같이 등록.

    이게 없으면 모델이 답을 다 하고도 안 멈추고, 다음 턴("Human: ...")을 새로
    지어내면서 프롬프트 전체를 다시 토해내는 경우가 있었음 (실제로 관측됨 —
    max_new_tokens을 다 채울 때까지 계속 생성해서 느려지고, 출력에 JSON이 아닌
    텍스트가 잔뜩 섞임).
    """
    tok = processor.tokenizer
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    for special in ["<|im_end|>", "<|endoftext|>"]:
        tid = tok.convert_tokens_to_ids(special)
        if tid is not None and tid != tok.unk_token_id:
            ids.add(tid)
    return list(ids) or None


@torch.inference_mode()
def run_chat(model, processor, contents, system_text="You are a helpful assistant.",
             do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=512,
             use_audio_in_video=False, fps=1.0, max_frames=32):
    """use_audio_in_video: contents에 {"type": "video", ...}가 있을 때, 영상에
    포함된 오디오 트랙도 같이 넣을지. fps/max_frames는 영상 프레임 샘플링 밀도
    (30초 클립을 다 넣으면 토큰/메모리가 크게 늘어나서 기본값을 보수적으로 잡음).
    """
    model_device = next(model.parameters()).device
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": contents},
    ]
    inputs = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, padding=True,
        # use_audio_in_video만으로는 오디오 자리표시 토큰만 만들어지고 실제
        # 오디오 데이터는 안 뽑힘(-> next(audio_lengths)에서 StopIteration).
        # load_audio_from_video=True를 같이 줘야 apply_chat_template이 video
        # content에서 오디오 트랙을 실제로 추출해서 audio 입력으로 채워줌.
        use_audio_in_video=use_audio_in_video, load_audio_from_video=use_audio_in_video,
        fps=fps, max_frames=max_frames,
    ).to(model_device)
    # generate()가 position id(rope) 계산에 쓸 수 있게, 전처리 때와 동일한
    # use_audio_in_video 값을 inputs dict에도 넣어줘야 함 (Qwen2.5-Omni 공식
    # 예제가 그렇게 함 — apply_chat_template에만 넘기면 반영이 안 됨).
    inputs["use_audio_in_video"] = use_audio_in_video
    out_ids = model.generate(
        **inputs,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        eos_token_id=_eos_token_ids(processor),
    )
    # generate()가 입력 프롬프트 토큰까지 포함해서 반환하므로, 새로 생성된
    # 부분만 디코딩한다 (프롬프트를 다시 잘라내는 문자열 매칭보다 안전함).
    input_len = inputs["input_ids"].shape[1]
    new_tokens = out_ids[:, input_len:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return _strip_assistant_header(text)


def _strip_assistant_header(t):
    for mk in ["\nassistant\n", "\nassistant:", "assistant\n", "assistant:"]:
        if mk in t:
            t = t.split(mk)[-1]
    return t.strip()


def _first_balanced_json_object(t):
    """첫 '{'부터 시작해서, 문자열 리터럴 안의 중괄호는 무시하고 depth를 세어
    정확히 짝이 맞는 '}'에서 끊는다.

    예전엔 t.find("{") ~ t.rfind("}")로 잘랐는데, 모델이 답변을 끝낸 뒤에도
    멈추지 않고 프롬프트를 통째로 다시 생성해버리면(관측된 케이스) 그 안에도
    "{...}"가 여러 개 섞여 있어서 마지막 '}'가 엉뚱한 곳을 가리켜 파싱이
    깨졌음. depth 매칭이 그 문제를 근본적으로 없앤다.
    """
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    return None


def parse_json_object(text):
    """모델 출력에서 JSON 객체 하나를 최대한 관대하게 뽑아낸다.

    ```json ... ``` 코드블록, 앞뒤에 설명(또는 모델이 멈추지 않고 이어붙인
    엉뚱한 텍스트)이 붙은 raw JSON 둘 다 처리.
    """
    t = _strip_assistant_header(text)
    t = t.replace("```json", "```").replace("```JSON", "```")

    if "```" in t:
        parts = t.split("```")
        for i in range(1, len(parts), 2):
            chunk = parts[i].strip()
            try:
                return json.loads(chunk)
            except Exception:
                pass

    candidate = _first_balanced_json_object(t)
    if candidate is not None:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("No valid JSON object found in model output.")

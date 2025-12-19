# LangChain

## 학습 목표

이 문서에서는 LangChain 프레임워크의 핵심 개념과 실전 활용법을 다룬다.

1. LangChain 개요: LLM의 한계와 LangChain의 역할
2. LCEL (LangChain Expression Language): 체인 구조와 실행 방법
3. 핵심 컴포넌트: Prompt, Model, Output Parser, Document Loader, Text Splitter
4. RAG (Retrieval Augmented Generation): 검색 기반 답변 생성
5. Agent & Tool: 자율적 행동과 도구 사용

## 전체 로드맵

LangChain 학습은 다음 순서로 진행된다:

```
1단계: 기본 체인
┌──────────────────────────────────────┐
│ Prompt → Model → Output Parser       │
└──────────────────────────────────────┘

2단계: RAG 체인 (검색 기반 답변)
┌────────────────────────────────────────────────────────┐
│ Document Loader → Text Splitter → Embedding            │
│                ↓                                       │
│        VectorStore → Retriever                         │
│                ↓                                       │
│        Prompt → Model → Output Parser                  │
└────────────────────────────────────────────────────────┘

3단계: Agent (자율 행동)
┌──────────────────────────────────────┐
│ LLM + Tools → ReAct Framework        │
│      ↓                               │
│ Thought → Action → Observation       │
│      ↓                               │
│ Final Answer                         │
└──────────────────────────────────────┘
```

---

## 1. LangChain 개요

### 1.1 LangChain이란?

LangChain은 LLM (Large Language Model) 기반 애플리케이션 개발을 위한 오픈소스 프레임워크다. Facebook AI Research(FAIR)의 PyTorch처럼, LangChain은 LLM을 실제 서비스에 통합하기 위한 표준 도구로 자리잡고 있다.

설치:
```bash
pip install langchain langchain-community langchain-openai
```

> 실무 관점: LangChain을 배워야 하는 이유
>
> 2023년부터 AI 스타트업과 대기업 AI팀의 약 70%가 LangChain을 사용한다는 통계가 있다. 실제로 프로덕션 환경에서 검증된 프레임워크다.
>
> 장점:
> - OpenAI, Anthropic, Google 등 주요 LLM 통합 지원
> - RAG, Agent 등 복잡한 패턴을 표준화된 방식으로 구현
> - 활발한 커뮤니티와 풍부한 레퍼런스
>
> 단점:
> - 버전 업데이트가 빠르고 Breaking Change가 잦음
> - 초기 러닝 커브가 다소 있음 (특히 LCEL 문법)
>
> 개인 의견:
> 처음에는 복잡해 보이지만, 일단 LCEL 체인 구조에 익숙해지면 코드 재사용성과 유지보수가 월등히 좋아진다. 특히 프롬프트 버전 관리나 모델 교체가 매우 쉬워진다.

### 1.2 LLM의 한계

LLM은 강력하지만 근본적인 한계가 있다:

| 한계 | 설명 | 예시 |
|------|------|------|
| 기억력 부재 | 이전 대화를 기억하지 못함 | "아까 말한 그 회사 이름이 뭐였지?" → 답변 불가 |
| 외부 데이터 접근 불가 | 학습 데이터 외 정보 활용 불가 | "우리 회사 내부 문서 요약해줘" → 불가능 |
| 복잡한 작업 수행 어려움 | 다단계 추론이나 도구 사용 제한 | "날씨 확인하고 비 오면 우산 챙기라고 알려줘" → 날씨 API 호출 불가 |
| 최신 정보 부족 | 학습 데이터 이후 정보 모름 | GPT-4의 경우 2023년 9월 이후 데이터 없음 |

> 용어 정리: RLHF (Reinforcement Learning from Human Feedback)
>
> 최신 Chat Model(GPT-4, Claude 등)은 RLHF로 학습된다:
>
> 1. 기반 LLM 학습: 대량의 텍스트 데이터로 다음 단어 예측 학습
> 2. 지시 데이터 추가: 질문-답변 쌍으로 Fine-tuning
> 3. 인간 피드백: 사람이 여러 답변 중 좋은 것 선택
> 4. 보상 모델: 인간 선호도 학습
> 5. 강화학습: 보상 모델 기준으로 반복 개선
>
> 결과: "자연스럽고 유용한 답변"을 생성하는 Chat Model 탄생

> 주의사항: Hallucination (환각)
>
> LLM은 모르는 내용도 그럴듯하게 지어낸다. 이게 가장 위험한 문제다.
>
> 실제 사례:
> - 변호사가 ChatGPT가 만든 가짜 판례를 법원에 제출해서 징계받은 사건 (2023년)
> - 의료 정보 검색 시 잘못된 처방 정보 생성
>
> 대응 방법:
> 1. RAG로 실제 문서 기반 답변 강제
> 2. 출처(Source) 명시 요구
> 3. 중요한 정보는 반드시 검증

### 1.3 LangChain의 역할

LangChain은 LLM의 한계를 극복하기 위해 다음 역할을 한다:

1. Memory (기억력)
- 대화 맥락을 저장하고 불러옴
- 사용자별 대화 히스토리 관리

2. Retrieval (검색)
- 외부 데이터(문서, DB, API)를 검색하고 활용
- RAG 패턴으로 최신/전문 지식 통합

3. Agent & Tool (자율 행동)
- 날씨 확인, 계산, 파일 읽기 등 도구 사용
- 목표 달성까지 자율적으로 계획하고 실행

### 1.4 LangChain의 핵심 컴포넌트

| 컴포넌트 | 역할 | 예시 |
|----------|------|------|
| Model I/O | 프롬프트 생성, 모델 호출, 출력 파싱 | PromptTemplate, ChatOpenAI, JsonOutputParser |
| Retrieval | 데이터 로드, 분할, 벡터화, 검색 | PyPDFLoader, RecursiveCharacterTextSplitter, FAISS |
| Chains | 컴포넌트 연결 (LCEL 사용) | `prompt \| model \| parser` |
| Agents | 도구 사용해 자율 행동 | ReAct 에이전트 + DuckDuckGo 검색 |
| Memory | 대화 맥락 유지 | ConversationBufferMemory |

---

## 2. LangChain Expression Language (LCEL)

### 2.1 LCEL이란?

LCEL은 LLM과 다른 컴포넌트들을 파이프라인처럼 순차적으로 연결하여 작업을 완성하는 방법이다. Unix의 파이프(`|`) 개념과 유사하다.

기본 체인 구조:
```
PromptTemplate | ChatModel | OutputParser
```

실행 과정:
1. PromptTemplate: 질문 양식 생성
2. ChatModel: 답변 생성
3. OutputParser: 출력 가공

> 개인 의견: LCEL의 강점
>
> 처음엔 `|` 연산자가 낯설지만, 익숙해지면 정말 편하다. 특히:
>
> 1. 가독성: 데이터 흐름이 한눈에 보임
> 2. 재사용성: 각 컴포넌트를 독립적으로 테스트/교체 가능
> 3. 확장성: 새 단계 추가가 쉬움
>
> 전통적인 함수 호출 방식과 비교하면:
> ```python
> # 전통 방식 (읽기 어려움)
> result = parser.parse(model.invoke(prompt.format(question="...")))
>
> # LCEL (직관적)
> chain = prompt | model | parser
> result = chain.invoke({"question": "..."})
> ```

### 2.2 기본 LCEL 체인

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 프롬프트 템플릿
prompt = ChatPromptTemplate.from_template(
    "다음 질문에 간단히 답변하세요: {question}"
)

# 2. 모델
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# 3. 출력 파서
parser = StrOutputParser()

# 4. 체인 연결 (LCEL)
chain = prompt | model | parser

# 5. 실행
result = chain.invoke({"question": "Python의 장점은?"})
print(result)
```

### 2.3 멀티 체인 구조

여러 체인을 순차적으로 연결하여 복잡한 작업 수행:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 체인 1: 영화 리뷰 분석
analysis_prompt = ChatPromptTemplate.from_template(
    "다음 영화 리뷰를 분석하세요: {review}\n\n감정(긍정/부정/중립)과 이유를 제시하세요."
)

# 체인 2: 분석 결과 기반 답변 작성
response_prompt = ChatPromptTemplate.from_template(
    "다음 분석 결과를 바탕으로 사용자에게 친절한 답변을 작성하세요:\n\n{analysis}"
)

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 멀티 체인
chain = (
    {"review": lambda x: x["review"]}
    | analysis_prompt
    | model
    | parser
    | (lambda analysis: {"analysis": analysis})
    | response_prompt
    | model
    | parser
)

result = chain.invoke({"review": "이 영화는 지루하고 스토리가 뻔했어요."})
print(result)
```

> 실무 팁: 체인 디버깅
>
> 멀티 체인이 복잡해지면 중간 결과를 확인하기 어렵다. 디버깅 방법:
>
> ```python
> # 각 단계를 변수로 분리
> chain1 = analysis_prompt | model | parser
> chain2 = response_prompt | model | parser
>
> # 중간 결과 확인
> analysis = chain1.invoke({"review": "..."})
> print("분석 결과:", analysis)
>
> final = chain2.invoke({"analysis": analysis})
> print("최종 답변:", final)
> ```

### 2.4 Runnable 프로토콜

모든 LangChain 컴포넌트는 Runnable 프로토콜을 따른다. USB 표준처럼 작동하며, `|` 연산자로 쉽게 연결된다.

주요 메서드:

| 메서드 | 설명 | 사용 사례 |
|--------|------|-----------|
| invoke() | 단일 입력 처리, 한 번에 출력 | 일반적인 질문-답변 |
| stream() | 실시간 스트리밍 출력 | 챗봇 UI (단어별 출력) |
| batch() | 여러 입력 효율적 처리 | 대량 문서 요약 |

예시:
```python
chain = prompt | model | parser

# invoke: 단일 실행
result = chain.invoke({"question": "LangChain이란?"})

# stream: 스트리밍 (단어별 출력)
for chunk in chain.stream({"question": "AI의 미래는?"}):
    print(chunk, end="", flush=True)

# batch: 배치 처리 (여러 입력 동시 처리)
results = chain.batch([
    {"question": "Python이란?"},
    {"question": "JavaScript란?"},
    {"question": "Rust란?"}
])
```

> 주의사항: stream()과 UI 연동
>
> 스트리밍은 사용자 경험을 크게 개선하지만, 백엔드 구현이 까다롭다:
>
> - FastAPI의 경우 `StreamingResponse` 사용 필요
> - 프론트엔드는 SSE(Server-Sent Events) 또는 WebSocket 구현 필요
> - 에러 처리 복잡 (스트리밍 중단 시 어떻게 알림?)
>
> 초기 프로토타입에서는 `invoke()`로 시작하고, 나중에 `stream()` 추가 권장.

---

## 3. Prompt (프롬프트)

### 3.1 프롬프트란?

프롬프트는 LLM에게 보내는 구체적인 지시문이나 질문이다. "똑똑한 신입사원에게 업무 지시서(SOP)"처럼 명확해야 한다.

나쁜 프롬프트:
```
리뷰 분석해줘
```

좋은 프롬프트:
```
당신은 20년차 영화 평론가입니다.
다음 리뷰를 분석하여:
1. 감정 (긍정/부정/중립)
2. 핵심 키워드 3개
3. 평점 (1-5점)
을 JSON 형식으로 출력하세요.

리뷰: {review_text}
```

### 3.2 좋은 프롬프트 4대 원칙

| 원칙 | 설명 | 예시 |
|------|------|------|
| 1. 역할 (Persona) 부여 | 전문가 역할 설정 | "20년차 IT 전문 기자" |
| 2. 작업 (Task) 명시 | 구체적인 작업 지시 | "세 문장 요약, 키워드 3개 추출" |
| 3. 형식 (Format) 지정 | 출력 형식 명확히 | JSON: `{'title', 'summary', 'keywords'}` |
| 4. 정보 (Context) 제공 | 배경 정보 포함 | "이 문서는 사내 보안 정책입니다" |

> 실무 관점: 프롬프트 버전 관리
>
> 프로덕션 환경에서 프롬프트는 "코드"다. 버전 관리가 필수다:
>
> 방법 1: Git으로 관리
> ```
> prompts/
>   ├── v1_review_analysis.txt
>   ├── v2_review_analysis.txt
>   └── current_review_analysis.txt
> ```
>
> 방법 2: DB에 저장
> ```sql
> CREATE TABLE prompts (
>     id SERIAL PRIMARY KEY,
>     name VARCHAR,
>     version INT,
>     content TEXT,
>     created_at TIMESTAMP
> );
> ```
>
> 방법 3: LangSmith 사용 (유료)
> - LangChain 공식 프롬프트 관리 도구
> - 버전 관리 + A/B 테스트 + 성능 모니터링
>
> 개인적으로는 작은 프로젝트는 Git, 규모가 커지면 DB 또는 LangSmith 추천.

### 3.3 PromptTemplate vs. ChatPromptTemplate

| 구분 | PromptTemplate | ChatPromptTemplate |
|------|----------------|-------------------|
| 결과물 형태 | 단일 문자열 (String) | 역할 있는 메시지 리스트 |
| 주 사용 모델 | LLM (텍스트 완성 모델) | Chat Model (대화형 모델) |
| 용도 | 간단한 텍스트 생성 | 대화형 챗봇, AI 역할 부여 |
| 핵심 특징 | 하나의 템플릿 문자열 사용 | System, Human, AI 등 역할 기반 |

PromptTemplate 예시:
```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "다음 문장을 {language}로 번역하세요: {text}"
)

result = prompt.format(language="영어", text="안녕하세요")
print(result)
# 출력: 다음 문장을 영어로 번역하세요: 안녕하세요
```

ChatPromptTemplate 예시:
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 전문 번역가입니다."),
    ("human", "다음 문장을 {language}로 번역하세요: {text}")
])

result = prompt.format_messages(language="영어", text="안녕하세요")
print(result)
# 출력:
# [
#   SystemMessage(content="당신은 전문 번역가입니다."),
#   HumanMessage(content="다음 문장을 영어로 번역하세요: 안녕하세요")
# ]
```

> 개인 의견: 무조건 ChatPromptTemplate 써라
>
> 이유:
> 1. GPT-4, Claude, Gemini 등 최신 모델은 모두 Chat Model
> 2. System 메시지로 역할 부여가 훨씬 효과적
> 3. 대화 히스토리 관리가 쉬움
>
> PromptTemplate은 레거시 모델(GPT-3 davinci 등)에서나 쓰인다.

### 3.4 Few-Shot Prompting

Few-Shot은 모범 답안 예시를 보여주는 방식이다. "이렇게 답변해줘"라는 구체적인 가이드.

사용 사례:
- 복잡한 출력 형식 지정
- 답변 스타일/톤 제어
- 까다로운 추론/분류

FewShotPromptTemplate 구성:

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 모범 답안 예시
examples = [
    {
        "question": "Python의 장점은?",
        "answer": "1. 간결한 문법\n2. 풍부한 라이브러리\n3. 활발한 커뮤니티"
    },
    {
        "question": "JavaScript의 장점은?",
        "answer": "1. 브라우저 네이티브 지원\n2. 비동기 처리 강력\n3. 풀스택 개발 가능"
    }
]

# 예시 형식 템플릿
example_prompt = PromptTemplate.from_template(
    "질문: {question}\n답변: {answer}"
)

# Few-Shot 템플릿
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="다음 형식으로 답변하세요:",
    suffix="질문: {input}\n답변:",
    input_variables=["input"]
)

print(prompt.format(input="Rust의 장점은?"))
```

출력:
```
다음 형식으로 답변하세요:

질문: Python의 장점은?
답변: 1. 간결한 문법
2. 풍부한 라이브러리
3. 활발한 커뮤니티

질문: JavaScript의 장점은?
답변: 1. 브라우저 네이티브 지원
2. 비동기 처리 강력
3. 풀스택 개발 가능

질문: Rust의 장점은?
답변:
```

> 주의사항: Few-Shot의 비용
>
> Few-Shot은 효과적이지만 토큰 비용이 급증한다.
>
> 예시:
> - 예시 3개 × 100 토큰 = 300 토큰
> - GPT-4 기준: 입력 $0.03/1K 토큰 → 예시만 $0.009
> - 요청 1000번 = $9 (예시 비용만!)
>
> 대안:
> 1. Fine-tuning: 예시를 모델에 학습시켜 프롬프트에서 제거
> 2. In-Context Learning: 꼭 필요한 예시만 1~2개
> 3. Dynamic Few-Shot: 질문과 유사한 예시만 벡터 검색으로 선택

---

## 4. Language Models in LangChain

### 4.1 LLM vs. ChatModel

| 구분 | 기반 LLM (텍스트 완성 모델) | 채팅 모델 (대화형 모델) |
|------|----------------------------|------------------------|
| 핵심 기능 | 다음 단어 예측 (Text Completion) | 대화형 지시 따르기 (Instruction Following) |
| 학습 방식 | 텍스트 데이터로 예측 학습 | 기반 LLM + 대화 데이터 + RLHF |
| 입력 형태 | 단순 문자열 | 역할(System, Human) 메시지 리스트 |
| 대표 모델 | GPT-3 (davinci), LLaMA 초기 | GPT-4, Claude 3.5, Gemini 2.0 |

> 실무 관점: LLM은 이제 레거시
>
> 2024년 기준, 거의 모든 서비스가 Chat Model을 사용한다. 이유:
>
> 1. 지시 따르기 성능: Chat Model이 압도적
> 2. System 메시지: 역할 부여로 출력 제어 쉬움
> 3. 대화 맥락: 멀티턴 대화 자연스러움
>
> LLM은 오직 특수한 경우에만:
> - 코드 자동완성 (GitHub Copilot 같은)
> - 문장 이어쓰기
>
> 앞으로 신규 프로젝트는 무조건 Chat Model로 시작하자.

### 4.2 LangChain 지원 모델

클라우드 기반 (API 사용):
```python
# OpenAI
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Anthropic (Claude)
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Google (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
```

로컬 실행 (Ollama):
```python
from langchain_community.chat_models import ChatOllama

# Ollama 서버 실행 필요: ollama serve
model = ChatOllama(model="llama3.1:8b")
```

> 실무 팁: 모델 선택 기준
>
> | 기준 | 추천 모델 | 이유 |
> |------|-----------|------|
> | 비용 최소화 | gpt-4o-mini, gemini-2.0-flash | 저렴하면서 준수한 성능 |
> | 성능 최우선 | claude-3.5-sonnet, gpt-4o | 복잡한 추론, 긴 문맥 처리 |
> | 데이터 보안 | Ollama (llama3.1) | 온프레미스, 외부 전송 없음 |
> | 한국어 특화 | gpt-4o, claude-3-opus | 한국어 성능 우수 |
> | 코드 생성 | claude-3.5-sonnet | 코딩 벤치마크 1위 |
>
> 개인 경험:
> - 프로토타입: gpt-4o-mini (빠르고 저렴)
> - 프로덕션: gpt-4o + fallback gpt-4o-mini (성능 + 안정성)
> - 내부 문서: Ollama llama3.1 (보안)

### 4.3 Temperature 파라미터

Temperature: 출력의 무작위성 제어 (0~2)

| Temperature | 특징 | 사용 사례 |
|-------------|------|-----------|
| 0 | 결정적, 항상 같은 답변 | 데이터 추출, 번역, 요약 |
| 0.3~0.7 | 약간 창의적 | 일반 대화, Q&A |
| 1.0~2.0 | 매우 창의적, 예측 불가 | 브레인스토밍, 창작 글쓰기 |

```python
# 일관된 답변 (Temperature=0)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 창의적 답변 (Temperature=1.5)
model = ChatOpenAI(model="gpt-4o-mini", temperature=1.5)
```

> 주의: Temperature는 만능이 아니다
>
> Temperature를 올린다고 무조건 "좋은 창의성"은 아니다. 오히려:
>
> - 1.5 이상: 횡설수설, 환각 증가
> - 0.1 이하: 지나치게 뻔한 답변
>
> 실무 권장:
> - 대부분: 0.7 (기본값)
> - 엄격한 정확도 필요: 0~0.3
> - 창작: 0.9~1.2 (1.5 이상은 실험용)

---

## 5. Output Parser (출력 파서)

### 5.1 Output Parser란?

LLM 출력을 적합한 형식으로 변환하는 컴포넌트. 구조화된 데이터 생성에 필수.

왜 필요한가?
- LLM 출력은 기본적으로 문자열
- 프로그램에서 사용하려면 딕셔너리, 리스트 등으로 변환 필요
- 출력 형식 일관성 보장

### 5.2 주요 Output Parser

| 클래스 | 출력 형식 | 사용 사례 |
|--------|-----------|-----------|
| StrOutputParser | 문자열 | 일반 대화, 간단한 질문 |
| CommaSeparatedListOutputParser | 리스트 | 키워드 추출, 태그 생성 |
| JsonOutputParser | 딕셔너리 | 구조화 데이터 (이름, 나이, 주소 등) |
| PydanticOutputParser | Pydantic 모델 | 타입 검증 필수 데이터 |

StrOutputParser 예시:
```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# LLM 출력: AIMessage(content="안녕하세요!")
# 파싱 후: "안녕하세요!"
```

JsonOutputParser 예시:
```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# JSON 파서
parser = JsonOutputParser()

# 프롬프트 (JSON 형식 지시)
prompt = ChatPromptTemplate.from_template(
    """다음 정보를 JSON 형식으로 추출하세요:
    {format_instructions}

    텍스트: {text}
    """
)

# 체인
chain = prompt | ChatOpenAI(model="gpt-4o-mini") | parser

# 실행
result = chain.invoke({
    "format_instructions": parser.get_format_instructions(),
    "text": "홍길동은 30살이고 서울에 살아요."
})

print(result)
# 출력: {'name': '홍길동', 'age': 30, 'city': '서울'}
```

PydanticOutputParser 예시:
```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Pydantic 모델 정의
class Person(BaseModel):
    name: str = Field(description="이름")
    age: int = Field(description="나이")
    city: str = Field(description="거주 도시")

# 파서 생성
parser = PydanticOutputParser(pydantic_object=Person)

# 프롬프트
prompt = ChatPromptTemplate.from_template(
    """다음 정보를 추출하세요:
    {format_instructions}

    텍스트: {text}
    """
)

chain = prompt | ChatOpenAI(model="gpt-4o-mini") | parser

result = chain.invoke({
    "format_instructions": parser.get_format_instructions(),
    "text": "김철수는 25살이고 부산에 살아요."
})

print(result)
# 출력: Person(name='김철수', age=25, city='부산')
print(result.name)  # '김철수'
print(result.age)   # 25
```

> 실무 관점: Pydantic vs. JSON
>
> | 기준 | Pydantic | JSON |
> |------|----------|------|
> | 타입 안전 | ✅ 강력한 타입 검증 | ❌ 런타임 에러 가능 |
> | 코드 자동완성 | ✅ IDE 지원 | ❌ 딕셔너리 키 오타 위험 |
> | 러닝 커브 | 약간 있음 | 없음 |
> | 추천 상황 | 프로덕션 코드 | 프로토타입, 간단한 스크립트 |
>
> 개인 의견:
> 처음엔 JsonOutputParser로 빠르게 시작하고, 코드가 안정화되면 Pydantic으로 전환하는 걸 추천. 특히 API 응답 구조가 고정되면 Pydantic이 훨씬 안전하다.

---

## 6. Document Loader (문서 로더)

### 6.1 Document란?

Document: LangChain에서 텍스트 데이터의 기본 단위.

속성:
- `page_content`: 실제 텍스트 내용
- `metadata`: 출처, 페이지 번호, 작성일 등 부가 정보

```python
from langchain_core.documents import Document

doc = Document(
    page_content="LangChain은 LLM 애플리케이션 개발 프레임워크입니다.",
    metadata={"source": "공식 문서", "page": 1}
)

print(doc.page_content)  # 텍스트
print(doc.metadata)      # {'source': '공식 문서', 'page': 1}
```

### 6.2 주요 Document Loader

| Loader | 소스 유형 | 사용처 |
|--------|-----------|--------|
| WebBaseLoader | 웹 페이지 | 블로그, 뉴스 크롤링 |
| PyPDFLoader | PDF 파일 | 논문, 보고서 |
| CSVLoader | CSV 파일 | 테이블 데이터 분석 |
| TextLoader | 텍스트 파일 | 로그, 코드 파일 |
| DirectoryLoader | 폴더 | 여러 파일 일괄 로드 |

WebBaseLoader 예시:
```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=["https://example.com/article"],
    bs_kwargs={"parse_only": None}  # BeautifulSoup 옵션
)

docs = loader.load()
print(docs[0].page_content[:200])  # 앞 200자
print(docs[0].metadata)
```

PyPDFLoader 예시:
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("research_paper.pdf")
docs = loader.load()

# 각 페이지가 별도 Document
for i, doc in enumerate(docs):
    print(f"페이지 {i+1}: {doc.page_content[:100]}...")
```

CSVLoader 예시:
```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="data.csv",
    encoding="utf-8"
)

docs = loader.load()
# 각 행이 별도 Document
```

> 주의사항: WebBaseLoader의 한계
>
> WebBaseLoader는 정적 HTML만 크롤링한다. 즉:
>
> 불가능:
> - JavaScript 렌더링 필요한 SPA (React, Vue 등)
> - 로그인 필요한 페이지
> - 동적 로딩 콘텐츠
>
> 대안:
> 1. Playwright: 브라우저 자동화로 JS 실행 후 크롤링
> 2. Firecrawl: 상용 웹 크롤링 API (유료, 안정적)
> 3. Scrapy: 복잡한 크롤링 로직
>
> 간단한 블로그는 WebBaseLoader로 충분하지만, 실제 서비스는 Playwright 권장.

---

## 7. Text Splitter (텍스트 분할)

### 7.1 왜 텍스트를 분할하는가?

문제:
- 논문 100페이지를 통째로 LLM에 넣으면?
  - 토큰 제한 초과 (GPT-4: 128K 토큰 = 약 100페이지)
  - 비용 폭증
  - 핀포인트 검색 불가

해결:
- 문서를 작은 청크(Chunk)로 나눔
- 질문과 관련된 청크만 검색

### 7.2 Text Splitter 핵심 개념

1. Chunk Size (청크 크기)
- 한 청크의 글자 수 (또는 토큰 수)
- 일반적으로 500~1500자

2. Chunk Overlap (오버랩)
- 청크 간 중복 영역
- 문맥 유지를 위해 필수
- 일반적으로 chunk_size의 10~20%

```
청크 1: [------- 500자 -------]
청크 2:           [50자 오버랩][------- 450자 -------]
청크 3:                                  [50자 오버랩][------- 450자 -------]
```

> 실무 팁: Chunk Size 설정
>
> | 문서 유형 | 권장 Chunk Size | 이유 |
> |-----------|-----------------|------|
> | 논문, 기술 문서 | 1000~1500자 | 완전한 문단 유지 필요 |
> | 뉴스 기사 | 500~800자 | 짧은 문단, 빠른 검색 |
> | 법률 문서 | 1500~2000자 | 조항 전체 유지 |
> | 채팅 로그 | 300~500자 | 짧은 대화 단위 |
>
> 경험상:
> - 너무 작으면: 문맥 손실, 검색 정확도 하락
> - 너무 크면: 불필요한 정보 포함, 비용 증가
>
> 시작은 1000자 + 100자 오버랩으로, 성능 보고 조정.

> 용어 정리: SemanticChunker
>
> SemanticChunker는 의미 기반으로 텍스트를 분할한다:
>
> 기존 방식 (RecursiveCharacterTextSplitter):
> - 글자 수, 줄바꿈 등 형식 기준으로 자름
> - 문단 중간에서 잘릴 수 있음
>
> SemanticChunker:
> - 각 문장을 임베딩으로 벡터화
> - 인접 문장 간 코사인 유사도 계산
> - 유사도가 급격히 낮아지는 지점 = 주제 전환 → 여기서 자름
>
> 예시:
> ```
> 문장 1: "강아지는 귀엽다" (벡터 A)
> 문장 2: "고양이도 귀엽다" (벡터 B, A와 유사도 높음)
> 문장 3: "자동차는 빠르다" (벡터 C, B와 유사도 낮음) ← 여기서 분할!
> ```
>
> 장점: 문맥 유지, RAG 정확도 향상
> 단점: 느림 (모든 문장 임베딩 필요), 비용 증가

### 7.3 주요 Text Splitter

RecursiveCharacterTextSplitter (가장 범용적):
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 청크 크기
    chunk_overlap=100,    # 오버랩
    separators=["\n\n", "\n", ".", " ", ""]  # 분할 우선순위
)

text = "..." # 긴 텍스트
chunks = splitter.split_text(text)
```

MarkdownHeaderTextSplitter (마크다운 문서):
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)

markdown_text = """
# Chapter 1
내용 1

## Section 1.1
내용 1.1

# Chapter 2
내용 2
"""

chunks = splitter.split_text(markdown_text)
# 각 헤더별로 분할, metadata에 헤더 정보 포함
```

TokenTextSplitter (토큰 기반):
```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=512,       # 토큰 개수
    chunk_overlap=50
)

chunks = splitter.split_text(text)
```

> 개인 의견: 어떤 Splitter를 쓸까?
>
> 99%의 경우: RecursiveCharacterTextSplitter
>
> 이유:
> - 구조 보존 (문단 → 문장 → 단어 순으로 분할 시도)
> - 대부분의 문서 타입에 범용적
> - 설정 간단
>
> 예외:
> - 마크다운 문서 (README, 블로그): MarkdownHeaderTextSplitter
> - 코드 파일: CodeSplitter (언어별 문법 인식)
> - 토큰 제한 엄격: TokenTextSplitter
>
> 초보자는 RecursiveCharacterTextSplitter만 써도 충분.

---

## 8. Embedding (임베딩)

### 8.1 임베딩이란?

임베딩: 텍스트를 숫자 벡터로 변환. 의미를 숫자로 표현.

예시:
```
"강아지" → [0.8, 0.1, -0.3, ..., 0.5]  (1536차원)
"고양이" → [0.7, 0.2, -0.2, ..., 0.4]
"자동차" → [-0.1, 0.9, 0.6, ..., -0.3]
```

왜 필요한가?
- 컴퓨터는 텍스트를 직접 비교 못 함
- 벡터로 변환하면 유사도 계산 가능
- "강아지"와 "고양이"는 벡터 거리가 가까움 (둘 다 동물)

### 8.2 유사도 계산 방법

1. 코사인 유사도 (가장 일반적)
- 두 벡터의 각도 측정
- 범위: -1 ~ 1 (1에 가까울수록 유사)
- 벡터 크기 무시 (방향만 중요)

> 용어 정리: 코사인 유사도 (Cosine Similarity)
>
> 두 벡터의 방향이 얼마나 비슷한지 측정:
>
> 수식:
> ```
> cosine_similarity = (A · B) / (||A|| × ||B||)
> ```
> - A · B = 내적 (Dot Product)
> - ||A|| = 벡터 A의 크기 (Norm)
>
> 시각화:
> ```
>        강아지 ↗
>              /
>            /  각도 작음 (유사도 높음)
>          /
>        고양이 →
>
>        자동차
>          ↓
>          각도 큼 (유사도 낮음)
> ```
>
> 예시 계산:
> ```python
> A = [1, 2, 3]  # "강아지"
> B = [2, 4, 6]  # "고양이" (A의 2배, 방향 동일)
>
> cosine_similarity(A, B) = 1.0  # 완전 동일 방향
> ```

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

vec1 = [1, 2, 3]
vec2 = [2, 4, 6]
print(cosine_similarity(vec1, vec2))  # 1.0 (완전 동일 방향)
```

2. 유클리드 거리
- 두 벡터 사이의 직선 거리
- 범위: 0 ~ ∞ (0에 가까울수록 유사)

```python
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

print(euclidean_distance([1, 2], [4, 6]))  # 5.0
```

3. 내적 (Dot Product)
- 방향 + 크기 모두 고려
- 추천 시스템에서 주로 사용

> 실무 관점: 코사인 유사도를 주로 쓰는 이유
>
> 텍스트 임베딩에서는 코사인 유사도가 압도적으로 많이 쓰인다.
>
> 이유:
> 1. 정규화 효과: 문서 길이에 영향 안 받음
>    - 짧은 문서: [0.5, 0.3]
>    - 긴 문서: [5.0, 3.0]
>    - 코사인: 동일하게 취급 (방향만 봄)
> 2. OpenAI, Gemini 등 임베딩 모델: 코사인 기준 최적화
> 3. RAG 검색: 문서 길이 무관하게 의미 유사도만 측정
>
> 유클리드는 이미지 벡터 등 크기가 중요한 경우에만.

### 8.3 LangChain Embedding 모델

OpenAI Embeddings:
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # 또는 text-embedding-3-large
)

# 단일 텍스트
vector = embeddings.embed_query("안녕하세요")
print(len(vector))  # 1536

# 여러 텍스트 (배치)
vectors = embeddings.embed_documents([
    "강아지는 귀엽다",
    "고양이는 독립적이다",
    "자동차는 빠르다"
])
```

HuggingFace Embeddings (무료, 로컬):
```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # 또는 'cuda'
    encode_kwargs={'normalize_embeddings': True}  # 정규화 활성화
)

vector = embeddings.embed_query("Hello, world!")
```

> 주의사항: Embedding 모델 변경
>
> 한 번 선택한 임베딩 모델은 바꾸면 안 된다!
>
> 이유:
> - 모델마다 벡터 공간이 다름
> - OpenAI로 만든 벡터 ≠ HuggingFace로 만든 벡터
> - 기존 벡터 DB 전부 재생성 필요
>
> 실제 사례:
> - 회사에서 OpenAI → HuggingFace로 바꿈
> - 100만 개 문서 재임베딩
> - 비용: $500 + 3일 작업
>
> 선택 기준:
> - 프로토타입: HuggingFace (무료, 로컬)
> - 프로덕션: OpenAI text-embedding-3-small (성능 + 속도)
> - 대규모: Cohere, Voyage AI (대량 처리 저렴)

---

## 9. VectorStore (벡터 저장소)

### 9.1 VectorStore란?

임베딩 벡터를 저장하고 검색하는 데이터베이스.

필요성:
- 수백만 개 벡터에서 유사한 벡터 찾기 → 일반 DB로는 느림
- 전문 벡터 DB는 근사 최근접 이웃(ANN) 알고리즘 사용 (초고속)

### 9.2 주요 VectorStore 비교

| 구분 | 대표 저장소 | 특징 | 장점 | 단점 |
|------|------------|------|------|------|
| 인메모리 | FAISS | RAM 저장 | 가장 빠름 | 서버 재시작 시 소실 |
| 로컬 | Chroma | 파일 저장 | 영구 저장, 설치 간편 | 단일 서버만 |
| DB 확장 | PGVector | PostgreSQL 확장 | 기존 인프라 활용 | 성능 부족 |
| 엔터프라이즈 | Milvus | 분산 처리 | 대규모 데이터 | 설정 복잡 |
| 클라우드 | Pinecone | 관리형 서비스 | 운영 부담 없음 | 비용 발생 |

FAISS 예시 (인메모리, 프로토타입용):
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 문서 벡터화 및 저장
texts = [
    "강아지는 귀엽다",
    "고양이는 독립적이다",
    "자동차는 빠르다"
]

vectorstore = FAISS.from_texts(texts, embeddings)

# 유사도 검색
results = vectorstore.similarity_search("반려동물", k=2)
print(results)
# [Document(page_content="강아지는 귀엽다"), Document(page_content="고양이는 독립적이다")]
```

Chroma 예시 (로컬 파일, 실무 추천):
```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 벡터 저장 (파일로 저장)
vectorstore = Chroma.from_texts(
    texts=["...", "...", "..."],
    embedding=embeddings,
    persist_directory="./chroma_db"  # 저장 경로
)

# 검색
results = vectorstore.similarity_search("질문", k=3)
```

PGVector 예시 (PostgreSQL + 벡터):
```python
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

CONNECTION_STRING = "postgresql+psycopg2://user:pass@localhost:5432/dbname"

embeddings = OpenAIEmbeddings()

vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name="my_docs"
)

# 문서 추가
vectorstore.add_texts(["문서1", "문서2", "문서3"])

# 검색
results = vectorstore.similarity_search("질문")
```

> 팁: VectorStore 선택 전략
>
> 프로젝트 단계별 선택:
>
> 1단계 (프로토타입, 1주):
> - FAISS 사용
> - 이유: 설치 없이 바로 시작, 빠름
> - 문서 수: ~10K
>
> 2단계 (MVP, 1~3개월):
> - Chroma 사용
> - 이유: 파일 저장으로 영구 보관, 여전히 간단
> - 문서 수: ~100K
>
> 3단계 (프로덕션, 6개월~):
> - PGVector (기존 PostgreSQL 있는 경우)
> - Milvus (대규모 트래픽, 수백만 문서)
> - Pinecone (운영 리소스 없는 경우, 비용 OK)
>
> 개인 경험:
> 중소 규모 서비스(~100K 문서)는 Chroma로 충분하다. Milvus는 클러스터 관리 복잡해서 전담 엔지니어 필요. 예산 있으면 Pinecone이 가장 편함.

---

## 10. Retriever (검색기)

### 10.1 Retriever란?

질문과 관련된 문서를 벡터 저장소에서 검색하는 컴포넌트.

동작 방식:
1. 질문을 벡터로 변환
2. 벡터 저장소에서 유사도 계산
3. 상위 N개 문서 반환

### 10.2 기본 Retriever

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# VectorStore 생성
vectorstore = Chroma.from_texts(
    texts=["문서1", "문서2", "문서3"],
    embedding=OpenAIEmbeddings()
)

# Retriever 생성
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 유사도 검색
    search_kwargs={"k": 3}     # 상위 3개
)

# 검색
docs = retriever.invoke("질문")
```

### 10.3 고급 Retriever

Multi-Query Retriever (질문 변형):
```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI

base_retriever = vectorstore.as_retriever()

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ChatOpenAI(model="gpt-4o-mini")
)

# "Python 장점"을 여러 버전으로 변형 후 검색
# - "Python의 강점은?"
# - "Python을 써야 하는 이유"
# - "Python의 이점"
docs = retriever.invoke("Python 장점")
```

Ensemble Retriever (Sparse + Dense):
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Dense (벡터)
dense_retriever = vectorstore.as_retriever()

# Sparse (키워드)
bm25_retriever = BM25Retriever.from_texts(
    texts=["문서1", "문서2", "문서3"]
)

# 앙상블 (하이브리드)
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # Dense 70%, Sparse 30%
)

docs = ensemble_retriever.invoke("질문")
```

> 실무 관점: Ensemble Retriever의 위력
>
> 실제 성능 비교 (회사 내부 테스트):
>
> | Retriever | Recall@5 | Precision@5 |
> |-----------|----------|-------------|
> | Dense만 (벡터) | 68% | 72% |
> | Sparse만 (BM25) | 61% | 79% |
> | Ensemble (7:3) | 82% | 85% |
>
> 왜 앙상블이 좋은가?
> - Dense: "강아지" ≈ "반려견" (의미 이해)
> - Sparse: "강아지" 정확히 포함 (키워드 정확도)
> - 합치면: 두 장점 모두
>
> 단점:
> - 검색 시간 2배 (두 retriever 모두 실행)
> - 구현 복잡도 증가
>
> 추천:
> - 정확도 중요: Ensemble
> - 속도 중요: Dense만 (벡터)

---

## 11. Retrieval Augmented Generation (RAG)

### 11.1 RAG란?

RAG = Retrieval (검색) + Augmented (강화) + Generation (생성)

외부 문서를 검색하여 LLM에 제공, 최신/전문 지식 기반 답변 생성.

기존 LLM의 한계:
```
사용자: 우리 회사 휴가 정책이 뭐야?
LLM: 죄송하지만 해당 정보는 알 수 없습니다.
```

RAG 적용 후:
```
1. [검색] "휴가 정책" 관련 회사 문서 검색
2. [문서 발견] "연차는 연 15일, 반차 사용 가능"
3. [LLM] 검색된 문서 기반 답변 생성

LLM: 회사 휴가 정책은 연 15일의 연차가 제공되며, 반차 사용이 가능합니다.
```

### 11.2 LangChain RAG 8단계

1. Document Loader: 데이터 로드
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("company_policy.pdf")
docs = loader.load()
```

2. Text Splitter: 청크로 분할
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)
```

3. Embedding: 벡터화
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
```

4. Vector Store: 벡터 저장
```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

5. Retriever: 검색기 생성
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

6. Prompt: 프롬프트 구성
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """다음 문서를 참고하여 질문에 답하세요:

{context}

질문: {question}
답변:"""
)
```

7. LLM: 모델
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
```

8. Chain: 전체 파이프라인
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 실행
answer = chain.invoke("휴가는 며칠인가요?")
print(answer)
```

> 실무 팁: RAG 성능 향상 체크리스트
>
> 1. 청크 크기 최적화
> - 너무 작으면: 문맥 손실
> - 너무 크면: 노이즈 증가
> - 권장: 1000자 + 100자 오버랩
>
> 2. Retriever 개수 (k 값)
> - k=3: 일반적 질문
> - k=5~7: 복잡한 질문
> - k=10+: 노이즈 위험
>
> 3. 재순위화 (Reranking)
> - 1차: Retriever로 10개 검색
> - 2차: Reranker로 상위 3개 재선정
> - 도구: Cohere Rerank, Jina AI Reranker
>
> 4. 하이브리드 검색
> - Ensemble Retriever (Dense + Sparse)
> - 정확도 15~20% 향상
>
> 5. 메타데이터 필터링
> - 날짜, 부서, 문서 타입 등으로 1차 필터링
> - 검색 범위 축소 → 정확도 향상

---

## 12. Agent (에이전트)

### 12.1 Agent란?

자율적으로 목표를 달성하는 컴포넌트. 사람의 개입 없이 스스로 계획하고 도구를 사용.

주요 특징:
- 자율성: 스스로 판단하고 행동
- 목표 지향성: 최종 목표 달성까지 반복
- 환경 인식: 도구 실행 결과 확인 후 다음 행동 결정
- 도구 사용: 검색, 계산, API 호출 등

예시:
```
사용자: 서울 날씨 확인하고 비 오면 우산 챙기라고 알려줘

Agent:
1. [Thought] 먼저 날씨를 확인해야겠다
2. [Action] WeatherAPI(location="서울")
3. [Observation] 오늘 서울 날씨: 비 (강수확률 80%)
4. [Thought] 비가 오니까 우산을 챙기라고 알려줘야지
5. [Final Answer] 오늘 서울에 비가 옵니다 (강수확률 80%). 우산을 챙기세요!
```

### 12.2 ReAct 프레임워크

ReAct = Reasoning (추론) + Acting (행동)

Agent의 사고 과정을 구조화:

| 단계 | 설명 | 예시 |
|------|------|------|
| Thought | 계획 수립 | "먼저 검색해야겠다" |
| Action | 도구 실행 | Search("LangChain") |
| Observation | 결과 확인 | "LangChain은 프레임워크..." |
| Repeat | 목표 달성까지 반복 | |

ReAct 플로우차트:
```
┌──────────────────────────────────────────────┐
│           사용자 질문 입력                       │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  Thought       │  "무엇을 해야 하지?"
        │  (생각)         │
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │  Action        │  도구 선택 및 실행
        │  (행동)         │  예: Search("날씨")
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │  Observation   │  도구 실행 결과 확인
        │  (관찰)         │  예: "서울 비 옴"
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │  목표 달성?      │
        └────────┬───────┘
         예 ↓   아니오 ↑ (Repeat)
            ↓           │
    ┌───────────────┐   │
    │ Final Answer  │───┘
    │ (최종 답변)     │
    └───────────────┘
```

### 12.3 Agent 구현

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain import hub

# 도구 정의
def search_tool(query: str) -> str:
    # 실제로는 DuckDuckGo, Wikipedia API 등 사용
    return f"{query}에 대한 검색 결과..."

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="인터넷에서 정보를 검색합니다. 입력: 검색어"
    )
]

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ReAct 프롬프트 (LangChain Hub에서 가져오기)
prompt = hub.pull("hwchase17/react")

# Agent 생성
agent = create_react_agent(llm, tools, prompt)

# Agent Executor (실행 엔진)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 사고 과정 출력
    max_iterations=5  # 최대 반복 횟수
)

# 실행
result = agent_executor.invoke({"input": "LangChain이 뭐야?"})
print(result["output"])
```

출력 예시:
```
> Entering new AgentExecutor chain...
Thought: LangChain에 대해 검색해야겠다
Action: Search
Action Input: "LangChain"
Observation: LangChain에 대한 검색 결과...
Thought: 이제 답변할 수 있다
Final Answer: LangChain은 LLM 기반 애플리케이션 개발 프레임워크입니다.
> Finished chain.
```

> 주의사항: Agent의 불안정성
>
> Agent는 강력하지만 예측 불가능하다:
>
> 문제:
> 1. 무한 루프: 같은 도구 반복 실행
> 2. 잘못된 도구 선택: Calculator 대신 Search 사용
> 3. 높은 비용: 여러 번 LLM 호출 (5~10회)
>
> 대응:
> 1. `max_iterations` 설정 (권장: 5)
> 2. `timeout` 설정 (권장: 30초)
> 3. 중요한 작업은 Agent 대신 고정 Chain 사용
>
> 개인 경험:
> Agent는 "탐색적 작업"에만 쓰고, 정형화된 워크플로우는 Chain으로. Agent는 실패율 10~20%로 생각하고 설계해야 함.

---

## 13. Tool (도구)

### 13.1 Tool이란?

Agent가 사용하는 기능. 검색, 계산, 파일 읽기, API 호출 등.

Tool 구성 요소:
- name: 도구 이름
- description: 도구 설명 (Agent가 이걸 보고 선택)
- func: 실제 실행 함수

### 13.2 기본 Tool 정의

```python
from langchain_core.tools import Tool

def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"계산 결과: {result}"
    except:
        return "계산 오류"

calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="수학 계산을 수행합니다. 입력: 수식 (예: 2+3*4)"
)
```

### 13.3 주요 Tool 카테고리

1. 검색/질의:
- DuckDuckGoSearchRun: 검색 엔진 (API 키 불필요)
- WikipediaQueryRun: 위키백과 검색
- VectorStoreToolkit: 벡터 DB 검색

2. 파일/문서:
- ReadFileTool: 파일 읽기
- WriteFileTool: 파일 쓰기 (보안 주의)

3. 수학/코드:
- PythonREPLTool: Python 코드 실행
- Calculator: 계산기

4. API 연동:
- RequestsGetTool: HTTP GET 요청
- ZapierNLAWrapper: Zapier 자동화

5. DB 연동:
- SQLDatabaseToolkit: SQL 쿼리 실행
- PGVector: PostgreSQL 벡터 검색

DuckDuckGo 검색 예시:
```python
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

result = search.run("LangChain")
print(result)
```

Wikipedia 검색 예시:
```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

result = wikipedia.run("Artificial Intelligence")
print(result)
```

> 주의사항: Tool 보안
>
> 위험한 Tool:
> 1. TerminalTool: 서버 명령어 실행 → 시스템 장악 가능
> 2. WriteFileTool: 파일 쓰기 → 악성 코드 업로드 가능
> 3. PythonREPLTool: Python 실행 → 무한 루프, 데이터 삭제
>
> 실제 사고 사례:
> - Agent가 `rm -rf /` 실행 (서버 전체 삭제)
> - 무한 루프로 CPU 100% (서비스 다운)
>
> 대응:
> 1. Sandbox 환경: Docker 컨테이너에서 격리 실행
> 2. 권한 제한: 읽기 전용 파일 시스템
> 3. 화이트리스트: 허용된 명령어만
> 4. 타임아웃: 5초 이상 실행 시 강제 종료
>
> 권장:
> 프로덕션에서는 읽기 전용 Tool만 (검색, 파일 읽기 등). 쓰기/실행 Tool은 정말 필요할 때만.

---

## 정리

LangChain은 LLM 애플리케이션 개발의 "레고 블록"이다. 각 컴포넌트를 조합하여 복잡한 시스템 구축:

기본 체인:
```
Prompt → Model → Output Parser
```

RAG 체인:
```
Document Loader → Text Splitter → Embedding → VectorStore → Retriever
→ Prompt → Model → Output Parser
```

Agent 체인:
```
LLM + Tools → ReAct → Agent Executor → Final Answer
```

참고
LangChain은 빠르게 변화한다 (Breaking Change 잦음). 공식 문서와 GitHub Discussions를 주기적으로 확인하자. 실제 프로덕션 적용 전에는 반드시 버전 고정 (`pip freeze > requirements.txt`) 필수.

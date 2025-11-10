# LangGraph

## 학습 목표

이 문서에서는 LangGraph를 활용한 고급 AI 워크플로우 설계와 구현을 다룬다.

1. LangGraph 개요: RAG의 한계와 그래프 기반 접근법
2. 핵심 개념: State, Node, Edge, Conditional Edge의 이해
3. Self-RAG: 검색 문서 관련성 평가로 Hallucination 감소
4. Corrective RAG: 질문 재작성과 반복 검색으로 정확도 향상
5. 실전 응용: 웹 검색 통합 RAG 시스템 구축

## 전체 로드맵

LangGraph 학습은 다음 순서로 진행된다:

1단계: RAG의 문제점 이해
- 고정된 파이프라인 → 유연성 부족
- Hallucination → 검증 불가
- 검색 실패 → 대안 없음

2단계: LangGraph 기본 개념
- State → Node → Edge
- Conditional Edge
- 순환 가능한 워크플로우

3단계: 고급 RAG 패턴
- Self-RAG: 관련성 평가
- Corrective RAG: 질문 재작성
- Web Search RAG: 외부 검색 통합

---

## 1. LangGraph란 무엇인가?

### 1.1 LangGraph 개요

LangGraph는 LLM 기반 애플리케이션의 흐름을 그래프 구조로 모델링하는 프레임워크다. LangChain 생태계의 일부로, 복잡한 AI 워크플로우를 Node, Edge, State로 설계하여 순환적이고 유연한 제어를 가능하게 한다.

설치:
```bash
pip install langgraph langchain langchain-openai
```

> 용어 정리: 그래프 (Graph)
>
> 컴퓨터 과학에서 그래프는 노드(Node)와 엣지(Edge)로 구성된 자료구조다:
>
> - Node (노드): 작업을 수행하는 단위 (예: 문서 검색, 답변 생성)
> - Edge (엣지): 노드 간 연결 (예: 검색 → 답변)
> - Directed Graph (방향 그래프): 엣지에 방향이 있음
> - Cyclic Graph (순환 그래프): 노드로 다시 돌아올 수 있음
>
> LangGraph의 특징:
> - 방향 그래프 (Directed)
> - 순환 가능 (Cyclic) ← 핵심
> - 조건부 분기 (Conditional)

### 1.2 LangGraph가 필요한 이유

LangChain의 기본 체인 구조는 단방향 파이프라인이다:

```
Document Loader → Text Splitter → Embedding → VectorStore
                                                ↓
                                            Retriever
                                                ↓
                                    Prompt → Model → Answer
```

문제점:

| 문제 | 설명 | 영향 |
|------|------|------|
| 고정된 흐름 | 한 번 실행 후 수정 불가 | 검색 실패 시 대응 불가 |
| 평가 불가 | 중간 결과 검증 없음 | Hallucination 방지 어려움 |
| 조건부 처리 부재 | if-else 로직 구현 어려움 | 상황별 대응 불가 |
| 재시도 불가 | 실패 시 처음부터 다시 시작 | 비효율적 |

> 💡 실무 관점: 기본 RAG의 한계
>
> 프로덕션 환경에서 기본 RAG를 운영하면서 겪는 실제 문제들:
>
> 사례 1: 검색 실패
> - 질문: "2023년 회사 매출은?"
> - 문서에는 "작년 매출 258조원" (2023이라는 키워드 없음)
> - 결과: 검색 실패 → "정보 없음" 답변
>
> 사례 2: 잘못된 검색
> - 질문: "파이썬 버전은?"
> - 검색: "파이썬" 키워드로 무관한 문서 검색
> - 결과: 잘못된 정보 기반 답변 (Hallucination)
>
> 사례 3: 불완전한 정보
> - 질문: "최신 AI 트렌드는?"
> - 내부 문서: 2023년까지만 있음
> - 결과: 구식 정보 제공 (웹 검색 필요)
>
> LangGraph 도입 후:
> - 검색 실패 시 질문 재작성 후 재검색
> - 검색 결과 관련성 평가 후 재시도
> - 내부 문서 부족 시 웹 검색 자동 통합
>
> 실제 정확도가 65% → 85%로 향상 (내부 테스트 결과)

### 1.3 LangGraph vs LangChain Agent

| 구분 | LangChain Agent | LangGraph |
|------|-----------------|-----------|
| 제어 방식 | LLM이 자율 결정 | 개발자가 명시적 정의 |
| 예측 가능성 | 낮음 (LLM 판단에 의존) | 높음 (고정된 로직) |
| 비용 | 높음 (반복적 LLM 호출) | 낮음 (필요 시만 호출) |
| 디버깅 | 어려움 | 쉬움 (각 노드 추적 가능) |
| 사용 사례 | 탐색적 작업, 실험 | 정형화된 워크플로우 |

예시: 문서 검색 실패 시 대응

```python
# LangChain Agent (LLM이 판단)
# → 예측 불가, 엉뚱한 도구 선택 가능

# LangGraph (명시적 정의)
def evaluate_documents(state):
    if state["relevance_score"] < 0.5:
        return "rewrite_query"  # 개발자가 정의한 로직
    return "generate_answer"
```

> 개인 의견: 언제 Agent를 쓰고 언제 LangGraph를 쓸까
>
> LangChain Agent 사용:
> - 작업이 비정형적이고 탐색적일 때
> - 예: "인터넷에서 정보 찾고 요약해줘" (범위가 넓음)
> - 실패해도 괜찮은 경우 (실험, 내부 도구)
>
> LangGraph 사용:
> - 워크플로우가 명확할 때
> - 예: "문서 검색 → 평가 → 재검색 or 답변" (단계가 정의됨)
> - 신뢰성이 중요한 경우 (고객 대응, 프로덕션)
>
> 실무에서는 80%는 LangGraph, 20%는 Agent를 사용한다. Agent는 "보험"으로 생각하자.

---

## 2. LangGraph 개념

### 2.1 State (상태)

State는 노드 간 정보를 전달하는 공유 메모리다. Python의 `TypedDict`로 정의하며, 각 노드는 State를 읽고 업데이트한다.

State 정의:
```python
from typing import TypedDict, Annotated
import operator

class RAGState(TypedDict):
    question: str              # 사용자 질문
    documents: list[str]       # 검색된 문서
    answer: str                # 생성된 답변
    relevance_score: float     # 관련성 점수
    retry_count: int           # 재시도 횟수
```

State 업데이트 방식:

| 방식 | 설명 | 코드 예시 |
|------|------|-----------|
| Overwrite (기본) | 값을 덮어씀 | `state["answer"] = "새 답변"` |
| Append (추가) | 리스트에 추가 | `Annotated[list, operator.add]` |
| Merge (병합) | 딕셔너리 병합 | `Annotated[dict, merge_dict]` |

Append 예시:
```python
from typing import Annotated
import operator

class State(TypedDict):
    # 기본: 마지막 값으로 덮어씀
    question: str

    # operator.add: 리스트에 누적
    documents: Annotated[list[str], operator.add]

# 사용 예시
# Node 1: documents = ["문서1"]
# Node 2: documents = ["문서2"]
# 최종 State: documents = ["문서1", "문서2"]
```

> ⚠️ 주의사항: State 크기 관리
>
> State는 매 노드마다 복사되므로 크기가 커지면 성능 저하:
>
> 문제:
> ```python
> class State(TypedDict):
>     documents: Annotated[list[str], operator.add]  # 계속 누적
>
> # 10개 노드 거치면 문서가 100개 이상 누적 가능
> ```
>
> 해결:
> 1. 필요한 정보만 저장
>    ```python
>    state["top_documents"] = documents[:3]  # 상위 3개만
>    ```
>
> 2. 중간 결과 삭제
>    ```python
>    def cleanup_node(state):
>        return {"documents": []}  # 초기화
>    ```
>
> 3. 외부 저장소 사용
>    ```python
>    # State에는 ID만 저장
>    state["document_ids"] = ["doc1", "doc2"]
>    # 실제 내용은 DB나 캐시에 저장
>    ```

### 2.2 Node (노드)

Node는 작업을 수행하는 함수다. State를 입력받아 처리 후 업데이트된 State를 반환한다.

Node 구조:
```python
def node_function(state: State) -> State:
    # 1. State에서 필요한 정보 읽기
    question = state["question"]

    # 2. 작업 수행 (DB 조회, API 호출, LLM 실행 등)
    result = perform_task(question)

    # 3. State 업데이트
    return {"answer": result}
```

실제 예시: 문서 검색 노드
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def retrieve_documents(state: RAGState) -> RAGState:
    """벡터 저장소에서 관련 문서 검색"""
    question = state["question"]

    # 벡터 저장소 검색
    vectorstore = FAISS.load_local("./db", OpenAIEmbeddings())
    docs = vectorstore.similarity_search(question, k=3)

    # State 업데이트
    return {
        "documents": [doc.page_content for doc in docs]
    }
```

> Node 설계 원칙
>
> 1. 단일 책임 (Single Responsibility)
> - 각 노드는 하나의 명확한 작업만 수행
> - 나쁜 예: `retrieve_and_answer_node` (검색 + 답변)
> - 좋은 예: `retrieve_node`, `answer_node` (분리)
>
> 2. 순수 함수 지향
> - 외부 상태 변경 최소화
> - 같은 입력이면 같은 출력
>
> 3. 에러 처리
> ```python
> def safe_retrieve_node(state):
>     try:
>         docs = vectorstore.search(state["question"])
>         return {"documents": docs}
>     except Exception as e:
>         return {"documents": [], "error": str(e)}
> ```
>
> 4. 로깅
> ```python
> import logging
>
> def retrieve_node(state):
>     logging.info(f"Retrieving for: {state['question']}")
>     docs = vectorstore.search(state["question"])
>     logging.info(f"Found {len(docs)} documents")
>     return {"documents": docs}
> ```

### 2.3 Edge (엣지)

Edge는 노드 간 연결을 정의한다. 다음에 실행할 노드를 지정한다.

기본 Edge:
```python
from langgraph.graph import StateGraph, END

graph = StateGraph(RAGState)

# 노드 추가
graph.add_node("retrieve", retrieve_documents)
graph.add_node("answer", generate_answer)

# Edge 추가 (retrieve → answer)
graph.add_edge("retrieve", "answer")

# 마지막 노드 → END
graph.add_edge("answer", END)
```

흐름:
```
retrieve → answer → END
```

### 2.4 Conditional Edge (조건부 엣지)

Conditional Edge는 조건에 따라 다음 노드를 선택한다. if-else 로직을 구현할 수 있다.

구조:
```python
def condition_function(state: State) -> str:
    """조건 판단 함수"""
    if state["relevance_score"] > 0.7:
        return "generate_answer"  # 노드 이름 반환
    else:
        return "rewrite_query"

# Conditional Edge 추가
graph.add_conditional_edges(
    "evaluate",              # 현재 노드
    condition_function,      # 조건 함수
    {
        "generate_answer": "answer",    # 조건 결과 → 노드 매핑
        "rewrite_query": "rewrite"
    }
)
```

흐름:
```
evaluate → (조건 판단)
            ├─ relevance_score > 0.7 → answer
            └─ relevance_score ≤ 0.7 → rewrite
```

실제 예시:
```python
def evaluate_relevance(state: RAGState) -> str:
    """검색 문서 관련성 평가"""
    relevance_score = state["relevance_score"]
    retry_count = state.get("retry_count", 0)

    if relevance_score > 0.7:
        return "good"
    elif retry_count >= 3:
        return "give_up"  # 3번 재시도 후 포기
    else:
        return "retry"

graph.add_conditional_edges(
    "evaluate",
    evaluate_relevance,
    {
        "good": "generate_answer",
        "retry": "rewrite_query",
        "give_up": END
    }
)
```

> 순환 그래프 (Cyclic Graph)
>
> LangGraph의 핵심 강점은 순환이다:
>
> 기존 LangChain (Acyclic):
> ```
> retrieve → answer → END
> (한 번 지나가면 끝)
> ```
>
> LangGraph (Cyclic):
> ```
> retrieve → evaluate → (관련성 낮음) → rewrite → retrieve
>     ↓                      ↑
>  (관련성 높음)              └─── 순환 가능!
>     ↓
>  answer → END
> ```
>
> 장점:
> - 재시도 로직 구현 가능
> - 품질 보장 (기준 충족까지 반복)
> - 복잡한 워크플로우 모델링
>
> 주의:
> - 무한 루프 방지 필수 (`retry_count` 등)
> - `recursion_limit` 설정 권장

---

## 3. LangGraph 구현 실습

### 3.1 그래프 생성 기본

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. State 정의
class RAGState(TypedDict):
    question: str
    answer: str

# 2. 노드 함수 정의
def question_node(state: RAGState) -> RAGState:
    return {"question": state["question"]}

def answer_node(state: RAGState) -> RAGState:
    answer = f"{state['question']}에 대한 답변입니다."
    return {"answer": answer}

# 3. 그래프 생성
graph = StateGraph(RAGState)

# 4. 노드 추가
graph.add_node("question", question_node)
graph.add_node("answer", answer_node)

# 5. Edge 추가
graph.set_entry_point("question")  # 시작 노드
graph.add_edge("question", "answer")
graph.add_edge("answer", END)

# 6. 컴파일
app = graph.compile()

# 7. 실행
result = app.invoke({"question": "LangGraph란?"})
print(result["answer"])
```

### 3.2 그래프 시각화

```python
from IPython.display import Image, display

# Mermaid 다이어그램 생성
display(Image(app.get_graph().draw_mermaid_png()))
```

출력:
```
question → answer → END
```

> 💡 실무 팁: 디버깅 도구
>
> 1. 상세 로그 출력:
> ```python
> result = app.invoke(
>     {"question": "LangGraph란?"},
>     config={"recursion_limit": 10}
> )
>
> # 각 노드 실행 과정 확인
> for step in app.stream({"question": "..."}):
>     print(step)
> ```
>
> 2. Checkpointer로 상태 추적:
> ```python
> from langgraph.checkpoint.memory import MemorySaver
>
> memory = MemorySaver()
> app = graph.compile(checkpointer=memory)
>
> # thread_id로 세션 관리
> config = {"configurable": {"thread_id": "user-123"}}
> result = app.invoke({"question": "..."}, config=config)
>
> # 이전 상태 조회 가능
> ```
>
> 3. LangSmith 연동 (유료):
> - 모든 노드 실행 기록 자동 저장
> - 웹 UI로 시각화
> - 성능 분석 및 비용 추적

### 3.3 실행 옵션

```python
# 기본 실행
result = app.invoke({"question": "..."})

# 스트리밍 실행 (단계별 결과)
for step in app.stream({"question": "..."}):
    print(step)

# 재귀 제한 설정
result = app.invoke(
    {"question": "..."},
    config={"recursion_limit": 10}  # 최대 10번 순환
)
```

---

## 4. Self-RAG: 관련성 평가로 Hallucination 감소

### 4.1 Self-RAG 개념

Self-RAG는 검색된 문서의 관련성을 평가하여, 관련 없는 문서 사용을 방지한다.

기존 RAG 문제:
```
질문: "LangGraph의 장점은?"
검색: ["LangChain 소개", "Python 기초", "RAG 개요"]
       ↑ 관련 없는 문서 포함

LLM: 관련 없는 정보까지 섞어서 답변 → Hallucination 발생
```

Self-RAG 해결:
```
질문: "LangGraph의 장점은?"
검색: ["LangChain 소개", "Python 기초", "RAG 개요"]
       ↓
평가: 각 문서 관련성 점수 계산
       ├─ "LangChain 소개": 0.3 (낮음) → 제거
       ├─ "Python 기초": 0.1 (낮음) → 제거
       └─ "RAG 개요": 0.8 (높음) → 사용

LLM: 관련 높은 문서만 사용 → 정확한 답변
```

### 4.2 Self-RAG 구현

State 정의:
```python
from typing import TypedDict

class SelfRAGState(TypedDict):
    question: str
    documents: list[str]
    relevance_scores: list[float]
    filtered_documents: list[str]
    answer: str
```

노드 정의:

1. 검색 노드:
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def retrieve_node(state: SelfRAGState) -> SelfRAGState:
    """문서 검색"""
    vectorstore = FAISS.load_local("./db", OpenAIEmbeddings())
    docs = vectorstore.similarity_search(state["question"], k=5)

    return {
        "documents": [doc.page_content for doc in docs]
    }
```

2. 관련성 평가 노드:
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def evaluate_relevance_node(state: SelfRAGState) -> SelfRAGState:
    """각 문서의 관련성 평가"""
    question = state["question"]
    documents = state["documents"]

    # LLM으로 관련성 점수 계산
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """다음 문서가 질문과 얼마나 관련있는지 0~1 점수로 평가하세요.

질문: {question}
문서: {document}

점수만 숫자로 출력하세요 (예: 0.8)"""
    )

    scores = []
    for doc in documents:
        result = llm.invoke(prompt.format(question=question, document=doc))
        score = float(result.content.strip())
        scores.append(score)

    # 점수 0.5 이상만 필터링
    filtered_docs = [
        doc for doc, score in zip(documents, scores)
        if score >= 0.5
    ]

    return {
        "relevance_scores": scores,
        "filtered_documents": filtered_docs
    }
```

3. 답변 생성 노드:
```python
def generate_answer_node(state: SelfRAGState) -> SelfRAGState:
    """필터링된 문서 기반 답변"""
    question = state["question"]
    documents = state["filtered_documents"]

    if not documents:
        return {"answer": "관련 정보를 찾을 수 없습니다."}

    context = "\n\n".join(documents)

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template(
        """다음 문서를 참고하여 질문에 답하세요:

{context}

질문: {question}
답변:"""
    )

    result = llm.invoke(prompt.format(context=context, question=question))

    return {"answer": result.content}
```

그래프 구성:
```python
from langgraph.graph import StateGraph, END

# 그래프 생성
graph = StateGraph(SelfRAGState)

# 노드 추가
graph.add_node("retrieve", retrieve_node)
graph.add_node("evaluate", evaluate_relevance_node)
graph.add_node("answer", generate_answer_node)

# Edge 추가
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "evaluate")
graph.add_edge("evaluate", "answer")
graph.add_edge("answer", END)

# 컴파일
app = graph.compile()

# 실행
result = app.invoke({"question": "LangGraph의 장점은?"})
print(result["answer"])
```

흐름도:
```
retrieve → evaluate → answer → END
           (관련성 평가)
```

> 💡 실무 관점: 관련성 평가 최적화
>
> LLM으로 각 문서를 평가하면 비용과 시간이 많이 든다:
>
> 문제:
> - 문서 10개 × LLM 호출 = 10번 API 호출
> - GPT-4 사용 시 $0.03/1K 토큰 × 10 = $0.3+ (한 번에!)
>
> 최적화 방법:
>
> 1. 배치 처리 (Batch Processing):
> ```python
> # 한 번에 모든 문서 평가
> prompt = """다음 문서들의 관련성을 각각 0~1 점수로 평가하세요:
>
> 질문: {question}
>
> 문서 1: {doc1}
> 문서 2: {doc2}
> ...
>
> JSON 형식으로 출력: [0.8, 0.3, 0.9, ...]
> """
> ```
>
> 2. Reranker 모델 사용:
> ```python
> from sentence_transformers import CrossEncoder
>
> # 무료 로컬 모델 (빠르고 저렴)
> reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
>
> # 점수 계산
> scores = reranker.predict([
>     (question, doc) for doc in documents
> ])
> ```
>
> 3. Cohere Rerank API (유료, 정확도 높음):
> ```python
> import cohere
>
> co = cohere.Client("YOUR_API_KEY")
> results = co.rerank(
>     query=question,
>     documents=documents,
>     top_n=3,
>     model="rerank-english-v2.0"
> )
> ```
>
> 개인 경험:
> - 프로토타입: LLM 평가 (정확하지만 느림)
> - 프로덕션: Reranker 모델 (80% 정확도, 10배 빠름)
> - 대규모: Cohere Rerank (95% 정확도, 중간 속도)

---

## 5. Corrective RAG: 질문 재작성으로 정확도 향상

### 5.1 Corrective RAG 개념

Corrective RAG는 검색 결과가 불만족스러울 때, 질문을 재작성하여 다시 검색한다.

문제 상황:
```
질문: "생성형AI 가우스를 만든 회사는?"
검색: "삼성전자", "LG전자", "네이버" (모호한 결과)
평가: 관련성 0.3 (낮음)
```

해결:
```
질문 재작성: "삼성 생성형AI 가우스 개발사"
재검색: "삼성전자가 2023년 개발한 생성형AI 가우스..."
평가: 관련성 0.9 (높음)
답변 생성: "삼성전자입니다."
```

### 5.2 Corrective RAG 구현

State 정의:
```python
class CorrectiveRAGState(TypedDict):
    question: str
    rewritten_question: str
    documents: list[str]
    relevance_score: float
    retry_count: int
    answer: str
```

노드 정의:

1. 질문 재작성 노드:
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def rewrite_query_node(state: CorrectiveRAGState) -> CorrectiveRAGState:
    """질문을 더 구체적으로 재작성"""
    question = state["question"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    prompt = ChatPromptTemplate.from_template(
        """다음 질문을 더 명확하고 구체적으로 재작성하세요.
        검색 엔진이 정확한 결과를 찾을 수 있도록 핵심 키워드를 포함하세요.

원래 질문: {question}

재작성된 질문:"""
    )

    result = llm.invoke(prompt.format(question=question))

    return {
        "rewritten_question": result.content,
        "retry_count": state.get("retry_count", 0) + 1
    }
```

2. 평가 조건 함수:
```python
def evaluate_and_decide(state: CorrectiveRAGState) -> str:
    """검색 결과 평가 후 다음 단계 결정"""
    relevance = state.get("relevance_score", 0)
    retry_count = state.get("retry_count", 0)

    if relevance >= 0.7:
        return "generate"  # 충분히 좋음 → 답변 생성
    elif retry_count >= 3:
        return "give_up"   # 3번 재시도 후 포기
    else:
        return "rewrite"   # 질문 재작성 후 재검색
```

전체 그래프:
```python
from langgraph.graph import StateGraph, END

graph = StateGraph(CorrectiveRAGState)

# 노드 추가
graph.add_node("retrieve", retrieve_node)
graph.add_node("evaluate", evaluate_relevance_simple)
graph.add_node("rewrite", rewrite_query_node)
graph.add_node("generate", generate_answer_node)

# Edge 추가
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "evaluate")

# Conditional Edge: 평가 결과에 따라 분기
graph.add_conditional_edges(
    "evaluate",
    evaluate_and_decide,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "give_up": END
    }
)

# 재작성 후 다시 검색 (순환!)
graph.add_edge("rewrite", "retrieve")
graph.add_edge("generate", END)

app = graph.compile()
```

흐름도:
```
retrieve → evaluate → (관련성 높음) → generate → END
              ↓              ↑
         (관련성 낮음)          │
              ↓              │
           rewrite ──────────┘
              (질문 재작성 후 재검색)
```

> ⚠️ 주의사항: 무한 루프 방지
>
> Corrective RAG는 순환 구조라 무한 루프 위험이 있다:
>
> 문제:
> ```python
> # retry_count 체크 없음
> def evaluate(state):
>     if state["relevance"] < 0.7:
>         return "rewrite"  # 계속 재시도 → 무한 루프!
> ```
>
> 해결 방법:
>
> 1. 재시도 횟수 제한:
> ```python
> MAX_RETRIES = 3
>
> def evaluate(state):
>     if state.get("retry_count", 0) >= MAX_RETRIES:
>         return "give_up"
>     if state["relevance"] < 0.7:
>         return "rewrite"
>     return "generate"
> ```
>
> 2. recursion_limit 설정:
> ```python
> result = app.invoke(
>     {"question": "..."},
>     config={"recursion_limit": 10}  # 최대 10번 노드 실행
> )
> ```
>
> 3. 타임아웃 설정:
> ```python
> import signal
>
> def timeout_handler(signum, frame):
>     raise TimeoutError("실행 시간 초과")
>
> signal.signal(signal.SIGALRM, timeout_handler)
> signal.alarm(30)  # 30초 제한
>
> try:
>     result = app.invoke({"question": "..."})
> finally:
>     signal.alarm(0)
> ```

---

## 6. Web Search RAG: 외부 검색 통합

### 6.1 Web Search RAG 개념

내부 문서에 정보가 없을 때, 웹 검색으로 보완한다.

시나리오:
```
질문: "생성형AI 가우스를 만든 회사의 2023년 매출은?"

1단계: 내부 PDF 검색
검색 결과: "생성형AI 가우스는 삼성전자 제품"
평가: 매출 정보 없음 (not_grounded)

2단계: 질문 재작성
재작성: "삼성전자 2023년 매출"

3단계: 웹 검색
검색 결과: "삼성전자 2023년 매출 258.94조원" (네이버 뉴스)

4단계: 최종 답변
"삼성전자의 2023년 매출은 258.94조원입니다."
```

### 6.2 Web Search RAG 구현

State 정의:
```python
class WebSearchRAGState(TypedDict):
    question: str
    rewritten_question: str
    documents: list[str]
    web_results: list[str]
    is_grounded: bool  # 문서에 답이 있는지
    answer: str
```

노드 정의:

1. Grounding 평가 노드:
```python
def evaluate_grounding_node(state: WebSearchRAGState) -> WebSearchRAGState:
    """답변이 문서에 근거하는지 평가"""
    question = state["question"]
    documents = state["documents"]

    # LLM으로 평가
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """다음 문서에 질문의 답이 포함되어 있나요?

문서:
{documents}

질문: {question}

답: yes 또는 no로만 답하세요."""
    )

    result = llm.invoke(prompt.format(
        documents="\n\n".join(documents),
        question=question
    ))

    is_grounded = "yes" in result.content.lower()

    return {"is_grounded": is_grounded}
```

2. 웹 검색 노드:
```python
from langchain_community.tools import DuckDuckGoSearchRun

def web_search_node(state: WebSearchRAGState) -> WebSearchRAGState:
    """웹에서 정보 검색"""
    query = state.get("rewritten_question", state["question"])

    # DuckDuckGo 검색
    search = DuckDuckGoSearchRun()
    results = search.run(query)

    return {"web_results": [results]}
```

3. 조건 함수:
```python
def decide_next_step(state: WebSearchRAGState) -> str:
    """문서 충분성 판단"""
    if state["is_grounded"]:
        return "generate_from_docs"  # 문서만으로 답변
    else:
        return "rewrite_and_search_web"  # 웹 검색 필요
```

전체 그래프:
```python
graph = StateGraph(WebSearchRAGState)

# 노드 추가
graph.add_node("retrieve", retrieve_node)
graph.add_node("evaluate_grounding", evaluate_grounding_node)
graph.add_node("rewrite", rewrite_query_node)
graph.add_node("web_search", web_search_node)
graph.add_node("generate_from_docs", generate_answer_node)
graph.add_node("generate_from_web", generate_answer_from_web_node)

# Edge
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "evaluate_grounding")

# Conditional Edge
graph.add_conditional_edges(
    "evaluate_grounding",
    decide_next_step,
    {
        "generate_from_docs": "generate_from_docs",
        "rewrite_and_search_web": "rewrite"
    }
)

graph.add_edge("rewrite", "web_search")
graph.add_edge("web_search", "generate_from_web")
graph.add_edge("generate_from_docs", END)
graph.add_edge("generate_from_web", END)

app = graph.compile()
```

흐름도:
```
retrieve → evaluate_grounding
              ├─ (grounded) → generate_from_docs → END
              └─ (not grounded) → rewrite → web_search → generate_from_web → END
```

> 💡 실무 팁: 웹 검색 도구 선택
>
> | 도구 | 장점 | 단점 | 추천 사용처 |
> |------|------|------|-------------|
> | DuckDuckGoSearchRun | 무료, API 키 불필요 | 검색 품질 낮음, 느림 | 프로토타입 |
> | Google Search API | 검색 품질 최고 | 유료 ($5/1K 쿼리) | 프로덕션 (예산 있음) |
> | Tavily API | AI 검색 특화, 구조화된 결과 | 유료 ($0.01/검색) | RAG 시스템 |
> | SerpAPI | 다양한 검색엔진 지원 | 유료 ($50/5K 쿼리) | 종합 검색 |
> | Bing Search API | Microsoft 생태계 통합 | 유료 ($3/1K 쿼리) | Azure 사용자 |
>
> 개인 경험:
> - 초기 개발: DuckDuckGo (무료)
> - MVP: Tavily (AI 특화, 결과 품질 좋음)
> - 대규모: Google Search API (신뢰도 최고)
>
> Tavily 사용 예시:
> ```python
> from langchain_community.tools.tavily_search import TavilySearchResults
>
> search = TavilySearchResults(max_results=3)
> results = search.invoke("삼성전자 2023 매출")
>
> # 구조화된 결과
> for result in results:
>     print(result["title"])
>     print(result["url"])
>     print(result["content"])
> ```

---

## 7. 실전 프로젝트: 하이브리드 RAG 시스템

### 7.1 프로젝트 개요

다음 기능을 모두 포함하는 프로덕션급 RAG 시스템:

- Self-RAG: 문서 관련성 평가
- Corrective RAG: 질문 재작성
- Web Search: 웹 검색 통합
- Human-in-the-Loop: 중요 판단 시 사용자 확인

### 7.2 전체 코드

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
import operator

# ========== State 정의 ==========
class HybridRAGState(TypedDict):
    question: str
    rewritten_question: str
    documents: Annotated[list[str], operator.add]
    web_results: list[str]
    relevance_score: float
    is_grounded: bool
    retry_count: int
    answer: str

# ========== 노드 정의 ==========
def retrieve_node(state: HybridRAGState):
    """1. 벡터 저장소에서 문서 검색"""
    question = state.get("rewritten_question") or state["question"]

    vectorstore = FAISS.load_local("./db", OpenAIEmbeddings())
    docs = vectorstore.similarity_search(question, k=3)

    return {"documents": [doc.page_content for doc in docs]}

def evaluate_relevance_node(state: HybridRAGState):
    """2. 문서 관련성 평가"""
    question = state["question"]
    documents = state["documents"]

    # 간단한 휴리스틱: 질문 키워드가 문서에 포함되는지
    keywords = question.lower().split()
    doc_text = " ".join(documents).lower()

    matches = sum(1 for kw in keywords if kw in doc_text)
    relevance_score = matches / len(keywords) if keywords else 0

    return {"relevance_score": relevance_score}

def rewrite_query_node(state: HybridRAGState):
    """3. 질문 재작성"""
    question = state["question"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "다음 질문을 더 구체적으로 재작성하세요: {question}"
    )

    result = llm.invoke(prompt.format(question=question))

    return {
        "rewritten_question": result.content,
        "retry_count": state.get("retry_count", 0) + 1
    }

def evaluate_grounding_node(state: HybridRAGState):
    """4. 문서에 답이 있는지 평가"""
    documents = state["documents"]
    question = state["question"]

    # LLM 평가
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """문서에 질문의 답이 있나요?

문서: {documents}
질문: {question}

답: yes 또는 no"""
    )

    result = llm.invoke(prompt.format(
        documents="\n".join(documents[:500]),  # 최대 500자
        question=question
    ))

    is_grounded = "yes" in result.content.lower()

    return {"is_grounded": is_grounded}

def web_search_node(state: HybridRAGState):
    """5. 웹 검색"""
    query = state.get("rewritten_question") or state["question"]

    search = DuckDuckGoSearchRun()
    results = search.run(query)

    return {"web_results": [results]}

def generate_answer_node(state: HybridRAGState):
    """6. 최종 답변 생성"""
    question = state["question"]
    documents = state["documents"]
    web_results = state.get("web_results", [])

    # 모든 정보 통합
    context = "\n\n".join(documents + web_results)

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template(
        """다음 정보를 참고하여 질문에 답하세요:

{context}

질문: {question}
답변:"""
    )

    result = llm.invoke(prompt.format(context=context, question=question))

    return {"answer": result.content}

# ========== 조건 함수 ==========
def decide_relevance(state: HybridRAGState) -> str:
    """문서 관련성 판단"""
    relevance = state["relevance_score"]
    retry_count = state.get("retry_count", 0)

    if relevance >= 0.5:
        return "evaluate_grounding"
    elif retry_count >= 2:
        return "web_search"
    else:
        return "rewrite"

def decide_grounding(state: HybridRAGState) -> str:
    """문서 충분성 판단"""
    if state["is_grounded"]:
        return "generate"
    else:
        return "rewrite_for_web"

# ========== 그래프 구성 ==========
graph = StateGraph(HybridRAGState)

# 노드 추가
graph.add_node("retrieve", retrieve_node)
graph.add_node("evaluate_relevance", evaluate_relevance_node)
graph.add_node("rewrite", rewrite_query_node)
graph.add_node("evaluate_grounding", evaluate_grounding_node)
graph.add_node("web_search", web_search_node)
graph.add_node("generate", generate_answer_node)

# Entry Point
graph.set_entry_point("retrieve")

# Edge
graph.add_edge("retrieve", "evaluate_relevance")

# Conditional Edge 1: 관련성 판단
graph.add_conditional_edges(
    "evaluate_relevance",
    decide_relevance,
    {
        "evaluate_grounding": "evaluate_grounding",
        "rewrite": "rewrite",
        "web_search": "web_search"
    }
)

# 재작성 후 재검색
graph.add_edge("rewrite", "retrieve")

# Conditional Edge 2: Grounding 판단
graph.add_conditional_edges(
    "evaluate_grounding",
    decide_grounding,
    {
        "generate": "generate",
        "rewrite_for_web": "rewrite"
    }
)

# 웹 검색 후 답변
graph.add_edge("web_search", "generate")
graph.add_edge("generate", END)

# 컴파일
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# ========== 실행 ==========
if __name__ == "__main__":
    result = app.invoke(
        {"question": "LangGraph의 주요 장점 3가지는?"},
        config={
            "configurable": {"thread_id": "demo-1"},
            "recursion_limit": 15
        }
    )

    print("\n=== 최종 답변 ===")
    print(result["answer"])

    print("\n=== 실행 통계 ===")
    print(f"재시도 횟수: {result.get('retry_count', 0)}")
    print(f"관련성 점수: {result.get('relevance_score', 0):.2f}")
    print(f"웹 검색 사용: {'예' if result.get('web_results') else '아니오'}")
```

### 7.3 실행 흐름 예시

Case 1: 이상적인 경우
```
retrieve → evaluate_relevance (0.8) → evaluate_grounding (yes) → generate → END
```

Case 2: 재작성 필요
```
retrieve → evaluate_relevance (0.3) → rewrite → retrieve → evaluate_relevance (0.7)
        → evaluate_grounding (yes) → generate → END
```

Case 3: 웹 검색 필요
```
retrieve → evaluate_relevance (0.7) → evaluate_grounding (no) → rewrite
        → retrieve → evaluate_relevance (0.4) → web_search → generate → END
```

> 프로덕션 배포 체크리스트
>
> 1. 에러 처리
> ```python
> def safe_node(func):
>     def wrapper(state):
>         try:
>             return func(state)
>         except Exception as e:
>             logging.error(f"Node {func.__name__} failed: {e}")
>             return {"error": str(e)}
>     return wrapper
>
> @safe_node
> def retrieve_node(state):
>     ...
> ```
>
> 2. 타임아웃 설정
> ```python
> from timeout_decorator import timeout
>
> @timeout(10)  # 10초 제한
> def web_search_node(state):
>     ...
> ```
>
> 3. 로깅 및 모니터링
> ```python
> import logging
>
> logging.basicConfig(level=logging.INFO)
>
> def retrieve_node(state):
>     logging.info(f"Retrieving for: {state['question']}")
>     # ... 로직 ...
>     logging.info(f"Found {len(docs)} documents")
> ```
>
> 4. 비용 추적
> ```python
> from langchain.callbacks import get_openai_callback
>
> with get_openai_callback() as cb:
>     result = app.invoke({"question": "..."})
>     print(f"비용: ${cb.total_cost:.4f}")
>     print(f"토큰: {cb.total_tokens}")
> ```
>
> 5. A/B 테스팅
> - 기존 RAG vs LangGraph 성능 비교
> - 정확도, 응답 시간, 비용 측정
> - 사용자 만족도 조사

---

## 8. 고급 주제

### 8.1 Human-in-the-Loop

중요한 판단이 필요한 경우, 사람이 개입할 수 있다.

```python
def human_approval_node(state: HybridRAGState):
    """사용자 확인 요청"""
    answer = state["answer"]

    print(f"\n생성된 답변:\n{answer}\n")
    approval = input("이 답변을 사용하시겠습니까? (y/n): ")

    if approval.lower() == 'y':
        return {"approved": True}
    else:
        return {"approved": False, "retry_count": state["retry_count"] + 1}

# Conditional Edge
def check_approval(state):
    if state.get("approved"):
        return "end"
    else:
        return "regenerate"

graph.add_node("human_approval", human_approval_node)
graph.add_edge("generate", "human_approval")
graph.add_conditional_edges(
    "human_approval",
    check_approval,
    {
        "end": END,
        "regenerate": "rewrite"
    }
)
```

### 8.2 Multi-Agent 협업

여러 Agent가 역할 분담하여 작업한다.

```python
class MultiAgentState(TypedDict):
    question: str
    research_result: str  # Researcher Agent
    review_result: str    # Reviewer Agent
    final_answer: str     # Writer Agent

# Researcher: 정보 수집
def researcher_node(state):
    # RAG 검색
    ...
    return {"research_result": result}

# Reviewer: 검증
def reviewer_node(state):
    # 팩트 체크
    ...
    return {"review_result": "approved"}

# Writer: 답변 작성
def writer_node(state):
    # 최종 답변
    ...
    return {"final_answer": answer}

graph.add_node("researcher", researcher_node)
graph.add_node("reviewer", reviewer_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "reviewer")
graph.add_edge("reviewer", "writer")
graph.add_edge("writer", END)
```

### 8.3 스트리밍 출력

실시간으로 진행 상황을 보여준다.

```python
# 스트리밍 실행
for step in app.stream({"question": "LangGraph란?"}):
    node_name = list(step.keys())[0]
    node_output = step[node_name]

    print(f"\n[{node_name}] 실행 중...")

    if "documents" in node_output:
        print(f"  → 문서 {len(node_output['documents'])}개 검색")

    if "relevance_score" in node_output:
        print(f"  → 관련성: {node_output['relevance_score']:.2f}")

    if "answer" in node_output:
        print(f"  → 답변: {node_output['answer'][:100]}...")
```

### 8.4 성능 최적화

1. 병렬 처리:
```python
# 여러 노드를 동시에 실행
graph.add_node("retrieve_pdf", retrieve_pdf_node)
graph.add_node("retrieve_web", retrieve_web_node)

# 두 노드 병렬 실행
graph.set_entry_point("retrieve_pdf")
graph.set_entry_point("retrieve_web")  # 동시 시작

# 결과 병합
graph.add_node("merge", merge_results_node)
graph.add_edge("retrieve_pdf", "merge")
graph.add_edge("retrieve_web", "merge")
```

2. 캐싱:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def retrieve_cached(question: str):
    """동일 질문은 캐시에서 반환"""
    return vectorstore.search(question)
```

---

## 9. 디버깅 및 트러블슈팅

### 9.1 일반적인 문제

문제 1: 무한 루프
```
증상: 프로그램이 멈추지 않음
원인: retry_count 체크 누락
해결: 최대 재시도 횟수 설정
```

문제 2: State 키 오류
```python
# 에러: KeyError: 'rewritten_question'
# 원인: 노드에서 생성하지 않은 키 참조

# 해결
def safe_get(state, key, default=None):
    return state.get(key, default)

question = safe_get(state, "rewritten_question", state["question"])
```

문제 3: LLM 호출 실패
```python
# Rate limit, timeout 등
def llm_call_with_retry(prompt, max_retries=3):
    for i in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if i == max_retries - 1:
                raise
            time.sleep(2  i)  # Exponential backoff
```

### 9.2 로그 분석

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('langgraph.log'),
        logging.StreamHandler()
    ]
)

def retrieve_node(state):
    logging.debug(f"State input: {state}")
    # ... 로직 ...
    logging.info(f"Retrieved {len(docs)} documents")
    logging.debug(f"State output: {updated_state}")
    return updated_state
```

---

## 정리

LangGraph는 복잡한 AI 워크플로우를 그래프로 모델링하여, 유연하고 제어 가능한 시스템을 구축할 수 있게 한다.

핵심 개념:
- State: 노드 간 공유 메모리
- Node: 작업 수행 함수
- Edge: 노드 연결
- Conditional Edge: 조건부 분기

고급 패턴:
- Self-RAG: 문서 관련성 평가
- Corrective RAG: 질문 재작성
- Web Search RAG: 웹 검색 통합
- Human-in-the-Loop: 사람 개입

실무 권장사항:
1. 무한 루프 방지 (retry_count, recursion_limit)
2. 에러 처리 및 로깅
3. 비용 추적 및 최적화
4. A/B 테스팅으로 효과 검증

LangGraph는 프로덕션급 RAG 시스템 구축에 필수 도구다. 기본 RAG로 시작하여, 점진적으로 Self-RAG, Corrective RAG를 추가하며 개선하자.

참고
LangGraph는 LangChain 생태계의 최신 도구로, 빠르게 업데이트된다. 공식 문서(https://langchain-ai.github.io/langgraph/)를 주기적으로 확인하자.

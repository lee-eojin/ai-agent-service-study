# Streamlit

## 학습 목표

이 문서에서는 Streamlit을 활용한 데이터 앱 개발의 핵심 개념과 실습을 다룬다.

1. Streamlit 개요: 웹 개발 지식 없이 파이썬만으로 앱 구축
2. 작동 원리: Top-Down 실행 모델과 상태 관리
3. 핵심 위젯: 텍스트, 데이터, 입력, 레이아웃 컴포넌트
4. AI/LLM 연동: 챗봇 UI와 스트리밍 출력
5. 실전 프로젝트: RAG 챗봇, 데이터 대시보드 구축

---

## 1. Streamlit 개요

### 1.1 Streamlit이란?

Streamlit은 데이터 과학자와 머신러닝 엔지니어를 위한 파이썬 웹 앱 프레임워크다. HTML, CSS, JavaScript 지식 없이 순수 파이썬만으로 대화형 웹 앱을 만들 수 있다.

설치:
```bash
pip install streamlit
```

첫 앱 실행:
```python
# app.py
import streamlit as st

st.title("첫 Streamlit 앱")
st.write("Hello, World!")
```

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 자동 실행

> 용어 정리: 포트 번호 8501
>
> Streamlit은 기본적으로 8501번 포트를 사용한다.
>
> 포트가 이미 사용 중인 경우:
> ```bash
> streamlit run app.py --server.port=8502
> ```
>
> 여러 앱 동시 실행:
> - 첫 번째 앱: 8501
> - 두 번째 앱: 8502 (자동 할당)
> - 세 번째 앱: 8503
>
> 방화벽 이슈:
> - macOS에서 처음 실행 시 방화벽 경고 → "허용" 클릭
> - 외부 접근 필요 시: `--server.address=0.0.0.0`

> 실무 관점: Streamlit을 배워야 하는 이유
>
> 2024년 기준, 데이터 과학 및 AI 프로토타이핑 분야에서 Streamlit은 사실상 표준이 되었다.
>
> 장점:
> - 데이터 분석 결과를 5분 만에 웹으로 공유
> - Pandas, Matplotlib, Plotly 등 데이터 라이브러리 네이티브 지원
> - LangChain, OpenAI 등 LLM 통합이 쉬움
>
> 단점:
> - 대규모 프로덕션 앱에는 부적합 (속도, 확장성)
> - 커스터마이징 제한 (디자인 자유도 낮음)
>
> 데이터 대시보드나 ML 데모를 빠르게 만들 때는 Streamlit이 최고다. 하지만 정식 서비스는 FastAPI + React 조합을 권장. Streamlit은 "아이디어 검증"과 "내부 도구" 용도로 적합하다.

### 1.2 Streamlit의 5가지 장점

| 장점 | 설명 | 예시 |
|------|------|------|
| 순수 파이썬 | HTML/CSS/JS 불필요 | `st.button("클릭")` 한 줄로 버튼 생성 |
| 놀라운 개발 속도 | 몇 줄로 앱 완성 | 10줄 코드로 데이터 대시보드 |
| 쉬운 UI 구성 | 위젯 함수 호출만으로 UI 완성 | 슬라이더, 파일 업로드 등 즉시 추가 |
| 완벽한 데이터 연동 | Pandas, NumPy, 시각화 라이브러리 호환 | `st.dataframe(df)` 로 인터랙티브 표 |
| AI/LLM 친화적 | 챗봇 UI, 스트리밍 출력 기본 제공 | LangChain 스트리밍 직접 지원 |

> 프로토타이핑 vs 프로덕션
>
> 프로토타이핑 (Prototyping):
> - 아이디어를 빠르게 검증하기 위한 초기 버전
> - 완벽함보다 속도 중시
> - Streamlit이 이상적
>
> 프로덕션 (Production):
> - 실제 사용자에게 제공하는 서비스
> - 성능, 보안, 확장성 필수
> - FastAPI/Django + React/Vue 조합
>
> 실무 흐름:
> 1. Streamlit으로 프로토타입 (1~2주)
> 2. 검증 후 프로덕션 재개발 (1~3개월)

### 1.3 Streamlit vs. 다른 프레임워크

| 구분 | Streamlit | Flask/FastAPI | Dash | Gradio |
|------|-----------|---------------|------|--------|
| 학습 곡선 | 매우 쉬움 | 중간 | 중간 | 쉬움 |
| 개발 속도 | 매우 빠름 | 느림 | 중간 | 빠름 |
| 커스터마이징 | 제한적 | 완전 자유 | 중간 | 제한적 |
| 데이터 과학 | 최적화됨 | 수동 구현 | 최적화됨 | ML 특화 |
| AI/LLM 통합 | 쉬움 | 수동 구현 | 어려움 | 매우 쉬움 |
| 프로덕션 적합도 | 낮음 | 높음 | 중간 | 낮음 |

> 프레임워크 선택 기준
>
> Streamlit 선택:
> - 내부 데이터 대시보드
> - ML 모델 데모
> - 빠른 POC(Proof of Concept)
>
> FastAPI/Flask 선택:
> - 외부 공개 서비스
> - 복잡한 비즈니스 로직
> - 모바일 앱 백엔드
>
> Gradio 선택:
> - 머신러닝 모델 즉시 공유 (HuggingFace 통합)
> - 이미지/오디오 입출력 중심

---

## 2. Streamlit의 작동 원리

### 2.1 Top-Down 실행 모델

Streamlit의 가장 큰 특징은 재실행 모델이다. 사용자가 위젯과 상호작용할 때마다 스크립트 전체가 위에서 아래로 다시 실행된다.

예시:
```python
import streamlit as st

st.title("카운터 앱")

# 이 코드는 버튼 클릭 시마다 재실행됨
count = 0
if st.button("증가"):
    count += 1

st.write(f"현재 카운트: {count}")
```

실행 결과:
- 버튼 클릭 시 count는 항상 0에서 시작 (재실행으로 초기화)
- 즉, count는 항상 0 또는 1만 표시

해결 방법: st.session_state 사용
```python
import streamlit as st

st.title("카운터 앱 (올바른 버전)")

# session_state로 상태 유지
if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("증가"):
    st.session_state.count += 1

st.write(f"현재 카운트: {st.session_state.count}")
```

> 초보자가 자주 하는 실수
>
> 문제 1: 변수가 계속 초기화됨
> ```python
> # 잘못된 코드
> counter = 0  # 재실행마다 0으로 초기화됨
> if st.button("증가"):
>     counter += 1
> ```
>
> 해결: session_state 사용 필수
>
> 문제 2: button의 리턴값 오해
> ```python
> # 잘못된 사용
> button_clicked = st.button("클릭")
> if button_clicked:
>     # 이 코드는 버튼 클릭한 그 순간만 실행됨
>     st.write("클릭됨")
> ```
>
> 버튼은 클릭한 순간에만 `True`, 재실행 시 `False`로 돌아간다.
> 상태 유지가 필요하면 session_state에 저장해야 한다.

> 용어 정리: 재실행 모델 (Rerun Model)
>
> 작동 방식:
> 1. 사용자가 앱 로드
> 2. 파이썬 스크립트가 위에서 아래로 실행
> 3. 위젯 상호작용 (버튼 클릭, 슬라이더 조작 등)
> 4. 전체 스크립트 재실행
> 5. 화면 업데이트
>
> 장점:
> - 상태 관리 단순 (콜백 함수 불필요)
> - 일반 Python 스크립트처럼 작성
>
> 단점:
> - 불필요한 재실행으로 성능 저하 가능
> - 대용량 데이터 로딩 시 느림
>
> 대응:
> - 캐싱(`@st.cache_data`, `@st.cache_resource`) 사용

### 2.2 캐싱으로 성능 최적화

문제 상황:
```python
import streamlit as st
import pandas as pd
import time

def load_data():
    time.sleep(5)  # 5초 걸리는 데이터 로딩
    return pd.read_csv("large_data.csv")

st.title("데이터 분석 대시보드")

# 매번 재실행 시 5초 대기 (매우 느림!)
df = load_data()

filter_value = st.slider("필터", 0, 100)
filtered_df = df[df["value"] > filter_value]
st.dataframe(filtered_df)
```

해결: @st.cache_data 사용
```python
import streamlit as st
import pandas as pd
import time

@st.cache_data  # 데이터 로딩 결과 캐싱
def load_data():
    time.sleep(5)
    return pd.read_csv("large_data.csv")

st.title("데이터 분석 대시보드")

# 첫 실행만 5초, 이후는 즉시 반환
df = load_data()

filter_value = st.slider("필터", 0, 100)
filtered_df = df[df["value"] > filter_value]
st.dataframe(filtered_df)
```

### 2.3 캐싱 종류

| 데코레이터 | 용도 | 예시 | 캐싱 방식 |
|-----------|------|------|-----------|
| @st.cache_data | 데이터 로딩 | CSV, API 호출, DataFrame | 값 복사 (안전) |
| @st.cache_resource | 모델/연결 | ML 모델, DB 연결, 객체 | 참조 공유 (빠름) |

@st.cache_data 예시:
```python
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

# 동일 파일 경로면 캐시 반환
df1 = load_csv("data.csv")  # CSV 로드 (느림)
df2 = load_csv("data.csv")  # 캐시 반환 (빠름)
```

@st.cache_resource 예시:
```python
@st.cache_resource
def load_model():
    from transformers import pipeline
    return pipeline("sentiment-analysis")

# 모델은 한 번만 로드 (메모리 절약)
model = load_model()
result = model("I love Streamlit!")
```

> 주의사항: 캐싱 오용
>
> 잘못된 사용:
> ```python
> @st.cache_data
> def get_current_time():
>     return datetime.now()  # 항상 최초 실행 시각 반환 (버그!)
> ```
>
> 올바른 사용:
> - 입력이 같으면 출력도 같은 함수만 캐싱
> - 예: 파일 읽기, 모델 로딩, API 호출 (동일 파라미터)
>
> 캐시 초기화 방법:
> - 브라우저에서 `C` 키 누르기
> - 코드에서 `load_data.clear()` 호출

---

## 3. Streamlit 핵심 위젯

### 3.1 텍스트 표시

```python
import streamlit as st

# 제목
st.title("메인 제목")

# 부제목
st.header("섹션 헤더")
st.subheader("서브 헤더")

# 일반 텍스트
st.text("고정폭 텍스트")

# Markdown
st.markdown("**볼드체**, *이탤릭*, `코드`")

# 만능 함수
st.write("텍스트, 변수, DataFrame 등 모든 것을 출력")
st.write("숫자:", 42)
st.write("리스트:", [1, 2, 3])
```

> st.write()의 동작 방식
>
> `st.write()`는 입력 타입에 따라 자동으로 최적 표시 방법 선택:
>
> | 입력 타입 | 표시 방법 |
> |-----------|-----------|
> | 문자열 | Markdown 렌더링 |
> | Pandas DataFrame | 인터랙티브 표 |
> | Matplotlib 그래프 | 이미지 |
> | Dict | JSON 형식 |
>
> 프로토타입에서는 `st.write()` 남발해도 괜찮다. 나중에 `st.dataframe()`, `st.pyplot()` 등으로 세밀하게 교체하면 된다.

### 3.2 데이터 표시

```python
import streamlit as st
import pandas as pd

# 샘플 데이터
df = pd.DataFrame({
    "이름": ["철수", "영희", "민수"],
    "나이": [25, 30, 28],
    "직업": ["개발자", "디자이너", "기획자"]
})

# 인터랙티브 표 (정렬, 검색 가능)
st.dataframe(df)

# 정적 표
st.table(df)

# 메트릭 (숫자 강조)
st.metric(label="총 사용자", value=1234, delta=56)
```

Matplotlib/Plotly 그래프:
```python
import matplotlib.pyplot as plt
import plotly.express as px

# Matplotlib
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
st.pyplot(fig)

# Plotly (인터랙티브)
fig = px.line(df, x="이름", y="나이")
st.plotly_chart(fig)
```

> 용어 정리: 인터랙티브 vs 정적
>
> 인터랙티브 (Interactive):
> - 사용자가 조작 가능 (확대, 정렬, 필터링)
> - 예: `st.dataframe()`, Plotly 차트
> - 장점: 사용자 경험 좋음
>
> 정적 (Static):
> - 고정된 이미지/표
> - 예: `st.table()`, Matplotlib 차트
> - 장점: 빠름, 인쇄 적합

### 3.3 미디어 표시

```python
import streamlit as st

# 이미지
st.image("cat.jpg", caption="귀여운 고양이", width=300)

# 오디오
st.audio("music.mp3")

# 비디오
st.video("demo.mp4")

# 외부 링크 (YouTube 등)
st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
```

---

### 3.4 입력 위젯 (기본)

```python
import streamlit as st

# 버튼
if st.button("클릭하세요"):
    st.write("버튼이 클릭되었습니다!")

# 체크박스
agree = st.checkbox("동의합니다")
if agree:
    st.write("감사합니다!")

# 라디오 버튼
genre = st.radio("좋아하는 장르", ["액션", "코미디", "드라마"])
st.write(f"선택: {genre}")

# 셀렉트박스
option = st.selectbox("언어 선택", ["Python", "JavaScript", "Go"])
st.write(f"선택한 언어: {option}")

# 멀티셀렉트
choices = st.multiselect("과일 선택", ["사과", "바나나", "오렌지"])
st.write(f"선택: {choices}")
```

슬라이더와 숫자 입력:
```python
# 슬라이더
age = st.slider("나이", 0, 100, 25)  # (레이블, 최소, 최대, 기본값)
st.write(f"선택한 나이: {age}")

# 범위 슬라이더
range_values = st.slider("가격 범위", 0, 1000, (200, 800))
st.write(f"범위: {range_values[0]} ~ {range_values[1]}")

# 숫자 직접 입력
number = st.number_input("수량", min_value=1, max_value=100, value=10)
st.write(f"수량: {number}")
```

> 슬라이더 vs number_input 선택 기준
>
> | 상황 | 추천 위젯 | 이유 |
> |------|-----------|------|
> | 범위가 명확한 경우 | `st.slider()` | 시각적으로 범위 파악 쉬움 |
> | 정확한 값 입력 | `st.number_input()` | 큰 숫자 입력 편리 |
> | 필터링 (연속값) | `st.slider()` | 실시간 변경 효과 확인 |
> | 설정값 (고정) | `st.number_input()` | 의도치 않은 변경 방지 |
>
> 조합 사용:
> ```python
> # 슬라이더 + 숫자 입력 동기화
> if "threshold" not in st.session_state:
>     st.session_state.threshold = 50
>
> col1, col2 = st.columns([3, 1])
> with col1:
>     st.session_state.threshold = st.slider("임계값", 0, 100, st.session_state.threshold)
> with col2:
>     st.session_state.threshold = st.number_input("값", 0, 100, st.session_state.threshold)
> ```

### 3.5 입력 위젯 (텍스트/파일)

```python
import streamlit as st

# 텍스트 입력
name = st.text_input("이름을 입력하세요")
st.write(f"안녕하세요, {name}님!")

# 텍스트 영역 (여러 줄)
message = st.text_area("메시지 입력", height=150)
st.write(f"입력한 메시지:\n{message}")

# 비밀번호 입력
password = st.text_input("비밀번호", type="password")

# 파일 업로드
uploaded_file = st.file_uploader("파일 선택", type=["csv", "txt", "pdf"])
if uploaded_file is not None:
    st.write(f"업로드된 파일: {uploaded_file.name}")

    # CSV 읽기 예시
    if uploaded_file.name.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
```

> 파일 업로드 시 주의사항
>
> 문제:
> - `uploaded_file`은 재실행 시 초기화됨 (사용자가 다시 업로드 필요)
> - 대용량 파일은 `st.session_state`에 저장
>
> ```python
> if uploaded_file is not None:
>     if "df" not in st.session_state:
>         st.session_state.df = pd.read_csv(uploaded_file)
>
> if "df" in st.session_state:
>     st.dataframe(st.session_state.df)
> ```

---

### 3.6 레이아웃

컬럼 (열 분할):
```python
import streamlit as st

col1, col2, col3 = st.columns(3)

with col1:
    st.header("컬럼 1")
    st.write("왼쪽 내용")

with col2:
    st.header("컬럼 2")
    st.write("중간 내용")

with col3:
    st.header("컬럼 3")
    st.write("오른쪽 내용")
```

사이드바:
```python
# 사이드바에 위젯 배치
st.sidebar.title("설정")
option = st.sidebar.selectbox("메뉴", ["홈", "데이터", "설정"])

# 메인 영역
st.title(f"{option} 페이지")
```

확장 가능한 컨테이너:
```python
with st.expander("자세히 보기"):
    st.write("이 부분은 클릭 시에만 표시됩니다.")
    st.dataframe(df)
```

탭:
```python
tab1, tab2, tab3 = st.tabs(["개요", "데이터", "차트"])

with tab1:
    st.header("개요")
    st.write("앱 설명...")

with tab2:
    st.header("데이터")
    st.dataframe(df)

with tab3:
    st.header("차트")
    st.line_chart(df)
```

---

### 3.7 AI/LLM 챗봇 UI

채팅 인터페이스:
```python
import streamlit as st

st.title("챗봇")

# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # AI 응답 (여기서는 간단히 반복)
    response = f"당신이 말한 '{prompt}'를 잘 받았습니다."
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
```

스트리밍 출력 (타이핑 효과):
```python
import streamlit as st
import time

def stream_response(text):
    """문자열을 한 글자씩 yield"""
    for char in text:
        yield char
        time.sleep(0.05)

st.title("스트리밍 챗봇")

if prompt := st.chat_input("질문 입력"):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response = "이것은 타이핑 효과가 적용된 응답입니다."
        st.write_stream(stream_response(response))
```

> LangChain 스트리밍 연동
>
> LangChain의 스트리밍 출력을 `st.write_stream()`에 직접 연결 가능:
>
> ```python
> from langchain_openai import ChatOpenAI
>
> model = ChatOpenAI(model="gpt-4o-mini", streaming=True)
>
> if prompt := st.chat_input("질문"):
>     with st.chat_message("assistant"):
>         st.write_stream(model.stream(prompt))
> ```
>
> 장점:
> - 사용자 경험 향상 (실시간 응답 느낌)
> - LLM 응답 대기 시간 체감 감소
>
> 주의:
> - 스트리밍 중 에러 처리 복잡
> - 로그 저장 시 전체 응답 조합 필요

---

### 3.8 상태 관리 (st.session_state)

기본 사용법:
```python
import streamlit as st

# 초기화
if "counter" not in st.session_state:
    st.session_state.counter = 0

# 증가
if st.button("증가"):
    st.session_state.counter += 1

# 감소
if st.button("감소"):
    st.session_state.counter -= 1

st.write(f"카운터: {st.session_state.counter}")
```

대화 기록 저장:
```python
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("메시지")
if st.button("전송"):
    st.session_state.chat_history.append({"user": user_input, "bot": f"응답: {user_input}"})

for chat in st.session_state.chat_history:
    st.write(f"**사용자:** {chat['user']}")
    st.write(f"**봇:** {chat['bot']}")
```

> 주의사항: session_state 초기화
>
> 문제:
> - 브라우저 새로고침 시 모든 상태 초기화
> - 여러 탭에서 열면 독립적인 세션
>
> 해결:
> 1. 로컬 파일 저장:
>    ```python
>    import json
>
>    # 저장
>    with open("state.json", "w") as f:
>        json.dump(st.session_state.to_dict(), f)
>
>    # 로드
>    with open("state.json", "r") as f:
>        st.session_state.update(json.load(f))
>    ```
>
> 2. 데이터베이스 연동:
>    - 사용자 ID별 세션 저장
>    - PostgreSQL, SQLite 등 활용

---

### 3.9 상태 표시 (st.status)

```python
import streamlit as st
import time

with st.status("데이터 처리 중...", expanded=True) as status:
    st.write("1단계: 데이터 로딩...")
    time.sleep(2)

    st.write("2단계: 전처리...")
    time.sleep(2)

    st.write("3단계: 분석...")
    time.sleep(2)

    status.update(label="완료!", state="complete", expanded=False)

st.write("처리가 완료되었습니다.")
```

진행률 표시:
```python
import streamlit as st
import time

progress_bar = st.progress(0)
status_text = st.empty()

for i in range(100):
    progress_bar.progress(i + 1)
    status_text.text(f"진행률: {i+1}%")
    time.sleep(0.05)

status_text.text("완료!")
```

---

## 4. 실전 프로젝트

### 4.1 데이터 대시보드

```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="매출 대시보드", layout="wide")

# 데이터 로드
@st.cache_data
def load_data():
    return pd.DataFrame({
        "날짜": pd.date_range("2024-01-01", periods=100),
        "매출": range(1000, 1100),
        "방문자": range(500, 600)
    })

df = load_data()

# 사이드바 필터
st.sidebar.header("필터")
date_range = st.sidebar.date_input(
    "기간 선택",
    value=(df["날짜"].min(), df["날짜"].max())
)

# 메인 대시보드
st.title("매출 대시보드")

# 메트릭 (3컬럼)
col1, col2, col3 = st.columns(3)
col1.metric("총 매출", f"{df['매출'].sum():,}원", "+12%")
col2.metric("평균 매출", f"{df['매출'].mean():.0f}원", "-3%")
col3.metric("총 방문자", f"{df['방문자'].sum():,}명", "+5%")

# 차트 (2컬럼)
col1, col2 = st.columns(2)

with col1:
    st.subheader("매출 추이")
    fig = px.line(df, x="날짜", y="매출")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("방문자 추이")
    fig = px.area(df, x="날짜", y="방문자")
    st.plotly_chart(fig, use_container_width=True)

# 데이터 테이블
with st.expander("원본 데이터 보기"):
    st.dataframe(df, use_container_width=True)
```

### 4.2 LangChain RAG 챗봇

```python
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

st.title("문서 기반 Q&A 챗봇")

# 사이드바: 문서 업로드
st.sidebar.header("문서 업로드")
uploaded_file = st.sidebar.file_uploader("텍스트 파일 선택", type=["txt"])

# VectorStore 초기화
if uploaded_file and "vectorstore" not in st.session_state:
    with st.status("문서 처리 중...", expanded=True) as status:
        # 1. 파일 읽기
        st.write("1. 파일 로딩...")
        content = uploaded_file.read().decode("utf-8")

        # 2. 청크 분할
        st.write("2. 텍스트 분할...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.create_documents([content])

        # 3. 임베딩 및 저장
        st.write("3. 벡터 저장소 생성...")
        embeddings = OpenAIEmbeddings()
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

        status.update(label="문서 처리 완료!", state="complete")

# 채팅 인터페이스
if "vectorstore" in st.session_state:
    # 채팅 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # RAG 체인 실행
        with st.chat_message("assistant"):
            # 1. 검색
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(prompt)
            context = "\n\n".join([doc.page_content for doc in docs])

            # 2. 프롬프트
            prompt_template = ChatPromptTemplate.from_template(
                """다음 문서를 참고하여 질문에 답하세요:

{context}

질문: {question}
답변:"""
            )

            # 3. LLM 호출 (스트리밍)
            model = ChatOpenAI(model="gpt-4o-mini", streaming=True)
            chain = prompt_template | model | StrOutputParser()

            response = st.write_stream(
                chain.stream({"context": context, "question": prompt})
            )

            # 응답 저장
            st.session_state.messages.append({"role": "assistant", "content": response})

            # 참고 문서 표시
            with st.expander("참고한 문서"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**청크 {i+1}:**")
                    st.text(doc.page_content[:200] + "...")

else:
    st.info("왼쪽 사이드바에서 문서를 업로드하세요.")
```

> RAG 성능 향상 방법
>
> 1. 청크 크기 최적화:
> - 문서 타입별 조정 (논문: 1500자, 뉴스: 500자)
> - A/B 테스트로 최적값 찾기
>
> 2. 재순위화 (Reranking):
> ```python
> from langchain.retrievers import ContextualCompressionRetriever
> from langchain.retrievers.document_compressors import CohereRerank
>
> compressor = CohereRerank()
> compression_retriever = ContextualCompressionRetriever(
>     base_compressor=compressor,
>     base_retriever=vectorstore.as_retriever()
> )
> ```
>
> 3. 하이브리드 검색:
> - Dense (벡터) + Sparse (BM25) 결합
> - Ensemble Retriever 사용

---

### 4.3 이미지 분류 앱

```python
import streamlit as st
from transformers import pipeline
from PIL import Image

st.title("이미지 분류 앱")

# 모델 로드 (캐싱)
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

model = load_model()

# 파일 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="업로드된 이미지", use_container_width=True)

    with col2:
        with st.spinner("분류 중..."):
            # 예측
            predictions = model(image)

            st.subheader("분류 결과")
            for pred in predictions[:5]:
                st.write(f"**{pred['label']}**: {pred['score']:.2%}")
                st.progress(pred['score'])
```

---

## 5. 배포

### 5.1 Streamlit Cloud (무료)

1. GitHub 저장소 생성:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/username/my-app.git
git push -u origin main
```

2. requirements.txt 작성:
```text
~~streamlit==1.30.0~~
~~pandas==2.1.0~~
~~plotly==5.18.0~~
~~langchain==0.1.0~~
~~langchain-openai==0.0.2~~
```
(2026.03.01 수정) 위 버전은 2024년 초 기준으로 구식이다. 실제 사용 시 `pip install streamlit langchain langchain-openai` 로 최신 버전 설치 후 `pip freeze > requirements.txt` 로 버전 고정 권장.

3. Streamlit Cloud 배포:
- https://share.streamlit.io 접속
- GitHub 저장소 연결
- 앱 경로 선택 (`app.py`)
- Deploy 클릭

장점:
- 완전 무료
- GitHub 푸시 시 자동 재배포
- HTTPS 기본 제공

단점:
- 공개 앱만 가능 (무료 플랜)
- 리소스 제한 (1GB RAM)
- 커스텀 도메인 불가

(2026.03.01 수정) Streamlit Community Cloud 정책이 변경될 수 있음. 배포 전 share.streamlit.io에서 최신 플랜 정책 확인 필요.

### 5.2 Docker 배포

Dockerfile:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

실행:
```bash
docker build -t my-streamlit-app .
docker run -p 8501:8501 my-streamlit-app
```

### 5.3 환경 변수 관리

secrets.toml (로컬):
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
DB_PASSWORD = "secret123"
```

코드에서 사용:
```python
import streamlit as st

api_key = st.secrets["OPENAI_API_KEY"]
db_password = st.secrets["DB_PASSWORD"]
```

Streamlit Cloud 설정:
- Dashboard → App Settings → Secrets
- secrets.toml 내용 붙여넣기

> 주의사항: API 키 보안
>
> 절대 하면 안 되는 것:
> - 코드에 API 키 하드코딩
> - GitHub에 secrets.toml 푸시
>
> .gitignore 필수:
> ```
> .streamlit/secrets.toml
> .env
> ```
>
> 환경별 관리:
> - 로컬: `.streamlit/secrets.toml`
> - 배포: Streamlit Cloud Secrets 또는 환경 변수

---

## 6. 성능 최적화

### 6.1 불필요한 재실행 방지

문제:
```python
import streamlit as st
import time

# 매번 실행됨 (느림!)
time.sleep(3)
st.write("로딩 완료")

name = st.text_input("이름")
```

해결:
```python
import streamlit as st
import time

@st.cache_data
def expensive_operation():
    time.sleep(3)
    return "로딩 완료"

result = expensive_operation()
st.write(result)

name = st.text_input("이름")
```

### 6.2 대용량 데이터 처리

Lazy Loading:
```python
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("large_file.csv")

st.title("데이터 뷰어")

if st.button("데이터 로드"):
    df = load_data()
    st.dataframe(df.head(100))  # 처음 100행만 표시
```

페이지네이션:
```python
@st.cache_data
def load_data():
    return pd.read_csv("large_file.csv")

df = load_data()
page_size = 50
page_num = st.number_input("페이지", min_value=1, max_value=len(df)//page_size)

start_idx = (page_num - 1) * page_size
end_idx = start_idx + page_size

st.dataframe(df.iloc[start_idx:end_idx])
```

---

참고: Streamlit은 빠르게 업데이트되므로 공식 문서(docs.streamlit.io)를 주기적으로 확인하자.

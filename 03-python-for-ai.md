# AI를 위한 Python

## 학습 목표

1. Python 개발 환경 구축: 가상환경 설정, VSCode 연동, 패키지 관리
2. Python 핵심 문법: 자료구조, 제어문, 함수, 객체지향, 예외 처리
3. AI 응용 실습: Flask 웹 서버, PostgreSQL 연동, LLM API 호출, 벡터 임베딩

---

## Part I: Python 기초 실습 및 환경 설정

### 1. Python 개발 환경 구성

AI 개발의 첫 단계는 독립적인 Python 실행 환경 구축이다. 프로젝트마다 서로 다른 패키지 버전을 사용할 수 있도록 가상환경을 설정한다.

#### 1.1 가상환경 생성 (venv)

macOS/Linux:
```bash
# 가상환경 생성
python3 -m venv myvenv

# 활성화
source myvenv/bin/activate

# 패키지 설치
pip install transformers flask openai

# 비활성화
deactivate
```

Windows:
```bash
python -m venv myvenv
myvenv\Scripts\activate
```

> 가상환경을 사용하는 이유
>
> 프로젝트 간 충돌 방지:
> - 프로젝트 A: Django 3.2 사용
> - 프로젝트 B: Django 4.0 사용
> - 시스템 Python에 둘 다 설치하면 충돌 발생
>
> 재현 가능한 환경:
> - `requirements.txt`로 정확한 패키지 버전 기록
> - 다른 팀원이나 서버에서 동일 환경 재현 가능
>
> 시스템 Python 보호:
> - macOS는 시스템 Python을 OS가 사용 (특히 `/usr/bin/python3`)
> - 실수로 시스템 패키지를 삭제하면 OS 문제 발생 가능
> - 가상환경으로 완전히 격리된 공간에서 작업

#### 1.2 Anaconda 환경 (선택)

데이터 과학 작업에는 Anaconda가 더 편리할 수 있다:

```bash
# Anaconda 설치 (macOS)
curl --output anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2024.10-1-MacOSX-x86_64.sh
bash anaconda.sh

# 환경 생성
conda create -n ai-env python=3.10

# 활성화
conda activate ai-env

# 패키지 설치
conda install numpy pandas matplotlib
pip install openai langchain

# 환경 목록 확인
conda info --envs

# 비활성화
conda deactivate
```

#### 1.3 VSCode 연동

1. VSCode 설치 (macOS):
```bash
# Homebrew로 설치
brew install --cask visual-studio-code

# 또는 https://code.visualstudio.com/download 에서 다운로드
```

2. Python 확장 설치:
- VSCode 실행 → Extensions (⌘+Shift+X) → "Python" 검색 및 설치

3. 인터프리터 선택:
- `⌘+Shift+P` → "Python: Select Interpreter"
- `./myvenv/bin/python` 선택

4. 테스트 코드 작성:
```python
# test.py
import sys
print(f"Python 버전: {sys.version}")
print(f"실행 경로: {sys.executable}")
```

5. 터미널에서 실행:
```bash
python test.py
```

---

### 2. Python 기본 문법

#### 2.1 변수와 데이터 타입

```python
# 기본 타입
age = 25              # int
height = 175.5        # float
name = "철수"         # str
is_student = True     # bool

# 타입 확인
print(type(age))      # <class 'int'>

# 타입 변환
str_age = str(age)    # "25"
int_height = int(height)  # 175
```

파이썬 특징:
- 동적 타이핑: 변수 선언 시 타입 지정 불필요
- 임의 정밀도: int는 메모리가 허용하는 한 무한대로 커질 수 있음
- 유니코드 지원: 문자열에서 한글, 이모지 등 자유롭게 사용

#### 2.2 연산자

산술 연산:
```python
10 / 2    # 5.0 (결과가 항상 float)
10 // 2   # 5 (정수 나눗셈)
10 % 3    # 1 (나머지)
2 ** 3    # 8 (거듭제곱)
```

비교 및 멤버십:
```python
x = [1, 2, 3]
1 in x        # True
4 not in x    # True

a = [1, 2]
b = [1, 2]
a == b        # True (값 비교)
a is b        # False (객체 비교)
```

비트 연산:
```python
# 8진수, 16진수, 2진수 표현
0o20   # 16 (8진수)
0x12   # 18 (16진수)
0b1010 # 10 (2진수)

# 비트 연산
5 & 3  # 1 (AND)
5 | 3  # 7 (OR)
5 ^ 3  # 6 (XOR)
~5     # -6 (NOT)
5 << 1 # 10 (왼쪽 시프트)
```

#### 2.3 문자열 처리

```python
# f-string (권장, Python 3.6+)
name = "Alice"
age = 30
print(f"{name}님은 {age}살입니다.")  # Alice님은 30살입니다.

# 문자열 메서드
text = "  Hello, Python!  "
text.strip()           # "Hello, Python!" (공백 제거)
text.split(",")        # ['  Hello', ' Python!  ']
"-".join(['a', 'b'])   # "a-b"
text.replace("Hello", "Hi")
text.find("Python")    # 9 (인덱스 반환, 없으면 -1)

# 슬라이싱
text = "Python"
text[0:3]   # "Pyt"
text[:3]    # "Pyt"
text[3:]    # "hon"
text[-1]    # "n" (뒤에서 첫 번째)
text[::-1]  # "nohtyP" (역순)
```

#### 2.4 입출력

```python
# 사용자 입력
name = input("이름을 입력하세요: ")
age = int(input("나이를 입력하세요: "))

# 포매팅 출력
print(f"{name}님은 {age}살입니다.")
```

---

### 3. 제어 구조

#### 3.1 조건문

```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"

# match-case (Python 3.10+)
match grade:
    case "A":
        print("우수")
    case "B":
        print("양호")
    case _:
        print("노력 필요")
```

> 들여쓰기의 중요성
>
> Python은 Java의 `{}`와 달리 콜론(`:`)과 들여쓰기로 코드 블록을 구분한다.
> - 들여쓰기는 보통 스페이스 4칸 권장 (PEP 8 표준)
> - VSCode는 자동으로 4칸 스페이스로 설정됨
> - 탭과 스페이스를 혼용하면 `IndentationError` 발생

#### 3.2 반복문

for 반복:
```python
# 기본 반복
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# enumerate (인덱스 포함)
fruits = ["apple", "banana", "cherry"]
for idx, fruit in enumerate(fruits):
    print(f"{idx}: {fruit}")

# for-else (break 없이 종료 시 실행)
for i in range(5):
    if i == 10:
        break
else:
    print("break 없이 완료")
```

리스트 컴프리헨션:
```python
# 일반 방식
squares = []
for x in range(10):
    squares.append(x**2)

# 컴프리헨션 (파이썬다운 방식)
squares = [x**2 for x in range(10)]

# 조건부 컴프리헨션
evens = [x for x in range(20) if x % 2 == 0]
```

---

### 4. 자료 구조

#### 4.1 리스트 (List)

```python
numbers = [1, 2, 3, 4, 5]

# 추가/삭제
numbers.append(6)      # [1, 2, 3, 4, 5, 6]
numbers.insert(0, 0)   # [0, 1, 2, 3, 4, 5, 6]
numbers.remove(3)      # 값 3 제거
numbers.pop()          # 마지막 요소 제거 및 반환

# 슬라이싱
numbers[1:3]    # [1, 2]
numbers[:3]     # 처음 3개
numbers[2:]     # 2번째부터 끝까지
numbers[:-1]    # 끝에 한 개만 제외
numbers[-2:]    # 뒤에서 2개
```

#### 4.2 튜플 (Tuple)

```python
# 불변(immutable) 리스트
point = (10, 20, 30, 40)
x, y, w, z = point  # 언패킹

# 튜플은 수정 불가
# point[0] = 15  # TypeError

# 리스트 ↔ 튜플 변환
numbers = [1, 2, 3]
mytuple = tuple(numbers)  # 리스트 → 튜플
mylist = list(point)      # 튜플 → 리스트
```

#### 4.3 집합 (Set)

```python
# 중복 제거 및 집합 연산
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

a | b  # {1, 2, 3, 4, 5, 6} (합집합)
a & b  # {3, 4} (교집합)
a - b  # {1, 2} (차집합)
```

#### 4.4 딕셔너리 (Dictionary)

```python
student = {
    "name": "태형",
    "age": 21,
    "major": "컴퓨터공학"
}

# 안전한 접근 (KeyError 방지)
print(student["name"])         # "태형"
print(student.get("age"))      # 21 (기본적으로 이걸 사용)
print(student.get("email", "없음"))  # "없음" 반환

# 추가/수정
student["grade"] = "A"  # 추가
student["age"] = 22     # 수정

# 순회
for key, value in student.items():
    print(f"{key}: {value}")
```

> 깊은 복사 vs 얕은 복사
>
> 얕은 복사 (Shallow Copy):
> ```python
> original = [[1, 2], [3, 4]]
> copied = original.copy()
> copied[0][0] = 99
> print(original)  # [[99, 2], [3, 4]] (원본도 변경됨!)
> ```
>
> 깊은 복사 (Deep Copy):
> ```python
> import copy
> original = [[1, 2], [3, 4]]
> copied = copy.deepcopy(original)
> copied[0][0] = 99
> print(original)  # [[1, 2], [3, 4]] (원본 유지)
> ```
>
> 언제 사용하는가?
> - 단순 값(int, str) 리스트: 얕은 복사로 충분
> - 중첩 구조(리스트 안의 리스트, 딕셔너리): 깊은 복사 필수
> - 리스트, 딕셔너리 등은 메모리 참조를 공유하므로 주의

---

### 5. 함수와 모듈화

#### 5.1 함수 정의

```python
def greet(name, message="안녕하세요"):
    return f"{message}, {name}님!"

print(greet("철수"))                # "안녕하세요, 철수님!"
print(greet("영희", "반갑습니다"))  # "반갑습니다, 영희님!"
```

#### 5.2 가변 인자

```python
# *args: 위치 인자를 튜플로 받음
def sum_all(*args):
    return sum(args)

sum_all(1, 2, 3, 4)  # 10

# **kwargs: 키워드 인자를 딕셔너리로 받음
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=30)
```

---

### 6. 객체지향 프로그래밍 (OOP)

#### 6.1 클래스 기본

```python
class Student:
    count = 0  # 클래스 변수 (모든 인스턴스 공유)

    def __init__(self, name, age):
        self.name = name  # 인스턴스 변수
        self.age = age
        Student.count += 1

    def introduce(self):
        return f"저는 {self.name}이고, {self.age}살입니다."

# 사용
s1 = Student("지민", 20)
s2 = Student("길동", 22)
print(Student.count)  # 2
print(s1.introduce())
```

#### 6.2 상속

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print("...소리를 낸다.")

class Dog(Animal):
    def speak(self):
        print(f"{self.name}가 멍멍 짖는다.")

class Cat(Animal):
    def speak(self):
        print(f"{self.name}가 야옹 운다.")

# 사용
dog = Dog("바둑이")
dog.speak()  # "바둑이가 멍멍 짖는다."
```

#### 6.3 캡슐화

```python
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        self.__balance = balance  # private 변수 (__)

    def set_deposit(self, amount):
        self.__balance += amount

    def set_withdraw(self, amount):
        if amount <= self.__balance:
            self.__balance -= amount
        else:
            print("잔액 부족")

    def get_balance(self):
        return self.__balance

# 사용
account = BankAccount("수지", 10000)
account.set_deposit(5000)
print(account.get_balance())  # 15000
# print(account.__balance)  # AttributeError
```

---

### 7. 파일 입출력

#### 7.1 텍스트 파일

```python
# 쓰기
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("안녕하세요\n")
    f.write("Python 파일 입출력\n")

# 읽기
with open("output.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(content)
```

#### 7.2 JSON 파일

```python
import json

# JSON 쓰기
data = {
    "name": "Alice",
    "age": 30,
    "skills": ["Python", "AI", "Docker"]
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# JSON 읽기
with open("data.json", "r", encoding="utf-8") as f:
    loaded_data = json.load(f)
    print(loaded_data["name"])
```

---

### 8. 예외 처리

```python
import traceback

try:
    num = int(input("숫자를 입력하세요: "))
    result = 10 / num
    print(result)
except ZeroDivisionError as e:
    print(f"0으로 나눌 수 없습니다: {e}")
except ValueError as e:
    print(f"숫자를 입력해야 합니다: {e}")
except Exception as e:
    print(f"예상치 못한 오류: {e}")
    traceback.print_exc()
finally:
    print("연산이 완료되었습니다.")
```

---

### 9. 네트워크 통신 기초

#### 9.1 Echo 서버 (Server)

```python
# echo_server.py
import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 9999))
server.listen(1)
print("서버 대기 중...")

conn, addr = server.accept()
print(f"연결됨: {addr}")

while True:
    data = conn.recv(1024).decode()
    if not data:
        break
    print(f"수신: {data}")
    conn.send(f"from server-> {data}".encode())

conn.close()
server.close()
```

#### 9.2 Echo 클라이언트 (Client)

```python
# echo_client.py
import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("localhost", 9999))

while True:
    msg = input("Enter message: ")
    if msg == "quit":
        break
    client.send(msg.encode())
    response = client.recv(1024).decode()
    print(f"Echo: {response}")

client.close()
```

---

## Part II: AI 응용 및 실전 연동

### 10. Flask 웹 애플리케이션 구축

#### 10.1 Flask 설치 및 기본 앱

```bash
pip install flask
```

```python
# app.py
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    html_out = """
    <html>
    <head>
        <title>Flask Example</title>
    </head>
    <body>
        <h1>Hello, Flask world!</h1>
        <p>This is a simple HTML page served by Flask.</p>
    </body>
    </html>
    """
    return html_out

@app.route("/user/<name>")
def user(name):
    return f"<h1>안녕하세요, {name}님!</h1>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

실행:
```bash
python app.py
```

브라우저에서 `http://localhost:5000` 접속

> macOS 방화벽 설정
>
> macOS에서 Flask 서버 실행 시 방화벽 경고가 뜰 수 있습니다.
> - 시스템 설정 → 네트워크 → 방화벽 → 방화벽 옵션
> - Python이나 터미널 앱에 대해 "들어오는 연결 허용" 선택
> - 또는 `host="127.0.0.1"`로 변경하여 로컬에서만 접속 가능하게 설정

#### 10.2 HTML 템플릿 사용

templates/index.html:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Flask App</title>
</head>
<body>
    <h1>{{ title }}</h1>
    <ul>
    {% for item in items %}
        <li>{{ item }}</li>
    {% endfor %}
    </ul>
</body>
</html>
```

app.py:
```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html",
                         title="AI 학습 목록",
                         items=["Python", "Docker", "PostgreSQL"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

---

### 11. PostgreSQL 연동

#### 11.1 패키지 설치 및 데이터베이스 연결

```bash
pip install psycopg2-binary
```

```python
import psycopg2

# 연결 (SSH 터널 사용 시 localhost)
db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'ai_dev',
    'user': 'aiuser',
    'password': 'securepassword'
}

conn = psycopg2.connect(**db_config)
cur = conn.cursor()
```

#### 11.2 테이블 생성 및 데이터 삽입

```python
# 테이블 생성
cur.execute("""
    CREATE TABLE IF NOT EXISTS mytest (
        id SERIAL PRIMARY KEY,
        title VARCHAR,
        doc_body VARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

# 데이터 삽입
sample_data = [
    ("첫 번째 제목", "이것은 첫 번째 문서의 본문 내용입니다."),
    ("두 번째 제목", "Python으로 PostgreSQL을 작성하는 예제입니다."),
    ("세 번째 제목", "데이터베이스 프로그래밍 테스트")
]

insert_query = """
    INSERT INTO mytest (title, doc_body)
    VALUES (%s, %s)
    RETURNING id;
"""

for title, doc_body in sample_data:
    cur.execute(insert_query, (title, doc_body))
    inserted_id = cur.fetchone()[0]
    print("inserted data id:", inserted_id)

conn.commit()
```

#### 11.3 데이터 조회

```python
# 최근 10개 데이터 조회
cur.execute("""
    SELECT id, title, doc_body, created_at
    FROM mytest
    ORDER BY id DESC
    LIMIT 10
""")

rows = cur.fetchall()
print("최근 10개 데이터:")
for row in rows:
    print(row)

cur.close()
conn.close()
```

---

### 12. Flask + PostgreSQL 통합

```python
from flask import Flask, render_template, request, g
import psycopg2

app = Flask(__name__)

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'ai_dev',
    'user': 'aiuser',
    'password': 'securepassword'
}

def get_db():
    """데이터베이스 연결 객체를 생성하고 반환"""
    if 'db' not in g:
        g.db = psycopg2.connect(**DB_CONFIG)
    return g.db

@app.teardown_appcontext
def close_db(error):
    """요청 종료 시 데이터베이스 연결을 닫음"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

@app.route("/")
def index():
    """mytest 테이블의 데이터를 페이지네이션하여 표시"""
    page = request.args.get('page', 1, type=int)
    per_page = 10

    try:
        conn = get_db()
        cur = conn.cursor()

        # 전체 항목 수 계산
        cur.execute("SELECT COUNT(*) FROM mytest;")
        total_items = cur.fetchone()[0]
        total_pages = (total_items + per_page - 1) // per_page

        # 현재 페이지 데이터 조회
        offset = (page - 1) * per_page
        query = """
            SELECT id, title, doc_body
            FROM mytest
            ORDER BY id DESC
            LIMIT %s OFFSET %s;
        """
        cur.execute(query, (per_page, offset))
        documents = cur.fetchall()

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        documents = []
        total_pages = 0
    finally:
        if 'cur' in locals():
            cur.close()

    return render_template("index.html",
                         documents=documents,
                         page=page,
                         total_pages=total_pages)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

---

### 13. LLM API 연동 (Gemini)

#### 13.1 API 키 발급

1. https://app.apidog.com 접속 및 가입
2. Gemini-2.0-flash-lite LLM 모델의 API 키 발급 (무료)

#### 13.2 Gemini API 호출

```bash
pip install openai
```

```python
# gemini_chat.py
from openai import OpenAI
import sys

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
# ~~LLM_ID = "gemini-2.0-flash-lite"~~
# (2026.03.01 수정) Gemini 모델명은 빠르게 업데이트됨. 아래 URL에서 최신 모델명 확인:
# https://ai.google.dev/gemini-api/docs/models/gemini
LLM_ID = "gemini-2.0-flash"

client = OpenAI(
    base_url=GEMINI_API_URL,
    api_key=GEMINI_API_KEY
)

def ai_chat(messages):
    print(f"GEMINI API 호출, MODEL={LLM_ID}")
    response = client.chat.completions.create(
        model=LLM_ID,
        messages=messages,
    )
    return response

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:", sys.argv[0], "질문을 입력하세요")
        sys.exit()

    question = ' '.join(sys.argv[1:])
    messages = [{"role": "user", "content": question}]

    response = ai_chat(messages=messages)
    print(response.choices[0].message.content)
```

실행:
```bash
python gemini_chat.py "서울시의 면적과 인구를 알려줘"
```

---

### 14. 벡터 임베딩 및 저장 (OpenAI)

#### 14.1 OpenAI API 설정

```bash
pip install openai
```

#### 14.2 텍스트 임베딩 생성

```python
# openai_vector_insert.py
import openai
import psycopg2

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
openai.api_key = OPENAI_API_KEY

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'ai_dev',
    'user': 'aiuser',
    'password': 'securepassword'
}

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """텍스트를 벡터로 변환"""
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def save_to_postgres(text: str, vector: list):
    """DB에 벡터 저장"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        sql = """
            INSERT INTO test_vector (text_str, text_vector)
            VALUES (%s, %s)
        """
        cur.execute(sql, (text, vector))
        conn.commit()
        print("데이터 저장 완료!")
    except Exception as e:
        print("DB 오류:", e)
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    input_text = input("임베딩할 문장을 입력하세요: ")
    embedding_vector = get_embedding(input_text)
    save_to_postgres(input_text, embedding_vector)
```

#### 14.3 유사도 검색

```python
# openai_vector_select.py
import openai
import psycopg2

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
openai.api_key = OPENAI_API_KEY

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'ai_dev',
    'user': 'aiuser',
    'password': 'securepassword'
}

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def find_most_similar(text: str):
    """가장 유사한 문장 검색"""
    vector = get_embedding(text)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        sql = """
            SELECT text_str, text_vector <-> %s AS distance
            FROM test_vector
            ORDER BY distance ASC
            LIMIT 1;
        """
        cur.execute(sql, (str(vector),))
        result = cur.fetchone()

        if result:
            matched_text, similarity = result
            print(f"\n가장 유사한 문장: {matched_text}")
            print(f"거리 (낮을수록 유사): {similarity:.6f}")
        else:
            print("유사한 문장을 찾을 수 없습니다.")
    except Exception as e:
        print("DB 오류:", e)
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    user_input = input("유사한 문장을 찾을 문장을 입력하세요: ")
    find_most_similar(user_input)
```

> 벡터 검색 연산자
>
> | 연산자 | 거리 측정 방식 | 사용 사례 |
> |--------|----------------|-----------|
> | `<->` | L2 거리 (유클리드) | 일반적인 거리 계산 |
> | `<=>` | 코사인 거리 | 텍스트 임베딩 검색 (가장 일반적) |
> | `<#>` | 내적 | 추천 시스템 |
>
> 코사인 유사도를 주로 사용하는 이유:
> - 벡터의 방향(의미)만 비교, 크기는 무시
> - 문서 길이에 영향을 받지 않음
> - OpenAI/Gemini 임베딩 모델이 코사인 유사도 기준으로 최적화됨


---
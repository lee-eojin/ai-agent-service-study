# 클라우드 기반 AI 개발 환경 구축

## 학습 목표

이 문서에서는 Oracle Cloud Infrastructure(OCI)를 활용하여 AI 개발 환경을 구축하는 과정을 다룬다.

1. **클라우드 서버 개념 이해**: VM과 Docker의 차이, 클라우드 네트워킹 기초
2. **OCI 서버 생성 및 설정**: Ubuntu 서버 생성, 방화벽 설정, SSH 접근 구성
3. **AI 개발 스택 구축**: PostgreSQL + pgVector, Python 환경, Docker 활용

---

## 1. VM과 Docker의 차이

### VM (Virtual Machine)

VM은 **하드웨어 수준의 가상화**를 제공한다. 물리 서버 위에 하이퍼바이저(Hypervisor)를 설치하고, 그 위에 완전히 독립된 운영체제를 실행한다.

**특징:**
- 각 VM은 자체 커널을 가진 완전한 OS
- 완벽한 격리 (한 VM의 문제가 다른 VM에 영향 없음)
- 부팅 시간이 길고 리소스 오버헤드가 큼
- OS 레벨의 라이브러리 버전 충돌 없음

**사용 사례:**
- 서로 다른 OS가 필요한 경우 (Windows + Linux)
- 완벽한 보안 격리가 필요한 경우
- 장기 운영되는 프로덕션 서버

### Docker (Container)

Docker는 **OS 수준의 가상화**를 제공한다. 호스트 OS의 커널을 공유하면서 프로세스와 파일시스템만 격리한다.

**특징:**
- 호스트와 동일한 커널 사용 (Linux 커널 공유)
- 가볍고 빠른 시작 (초 단위)
- 이미지 레이어 시스템으로 효율적 저장
- 포트 매핑으로 호스트와 통신

**사용 사례:**
- 마이크로서비스 아키텍처
- 개발/테스트 환경 표준화
- CI/CD 파이프라인
- 일시적인 작업 (데이터 처리, 배치 작업)

### 우리의 선택

이 프로젝트에서는:
- **OCI VM**: Ubuntu 22.04 서버를 VM으로 생성 (장기 운영, 안정성)
- **Docker**: PostgreSQL, nginx 등을 컨테이너로 실행 (쉬운 관리, 버전 관리)

---

## 2. OCI 서버 생성

### 2.1 인스턴스 생성

1. OCI 콘솔 접속 → Compute → Instances → Create Instance
2. 기본 설정:
   - **이름**: `ai-dev-server`
   - **이미지**: Ubuntu 22.04 Minimal
   - **Shape**: VM.Standard.E2.1.Micro (Always Free)
3. SSH 키 생성:
   - macOS 터미널에서 실행:
   ```bash
   # ED25519 알고리즘 사용 (RSA보다 빠르고 안전)
   ssh-keygen -t ed25519 -f ~/.ssh/oci_ai_key -C "oci-ai-server"
   ```
   - 생성된 공개키(`~/.ssh/oci_ai_key.pub`) 내용을 OCI에 붙여넣기

### 2.2 네트워크 설정 (VCN)

**VCN (Virtual Cloud Network)**: 클라우드 내부의 논리적 네트워크. 서브넷, 라우팅, 방화벽 규칙을 관리한다.

#### Subnet 구성

- **Public Subnet**: 인터넷 게이트웨이 연결, 공인 IP 할당 가능
- **Private Subnet**: 외부 접근 불가, 내부 서비스용

이번 실습에서는 Public Subnet을 사용해 SSH와 웹 접근을 허용한다.

#### Security List (방화벽 규칙)

| Port | Protocol | 용도 | 설정 방법 |
|------|----------|------|-----------|
| 22 | TCP | SSH | Ingress Rule 추가 (Source: 0.0.0.0/0) |
| 80 | TCP | HTTP | Ingress Rule 추가 (Source: 0.0.0.0/0) |
| 443 | TCP | HTTPS | Ingress Rule 추가 (Source: 0.0.0.0/0) |
| 5432 | TCP | PostgreSQL | **로컬에서만 접근** (SSH 터널 사용) |

**보안 주의사항:**
- PostgreSQL 포트(5432)는 Security List에 추가하지 않음
- 대신 SSH 터널링으로 안전하게 접근:
  ```bash
  ssh -i ~/.ssh/oci_ai_key -L 5432:localhost:5432 ubuntu@<서버IP>
  ```

---

## 3. macOS 개발 환경 설정

### 3.1 SSH 접속 설정

매번 긴 명령어를 치는 대신 `~/.ssh/config` 파일을 설정한다:

```bash
# ~/.ssh/config
Host oci-ai
    HostName <서버 공인 IP>
    User ubuntu
    IdentityFile ~/.ssh/oci_ai_key
    ServerAliveInterval 60
```

이제 간단하게 접속 가능:
```bash
ssh oci-ai
```

### 3.2 파일 전송

**scp (단일 파일):**
```bash
# 로컬 → 서버
scp -i ~/.ssh/oci_ai_key local_file.txt ubuntu@<서버IP>:/home/ubuntu/

# 서버 → 로컬
scp -i ~/.ssh/oci_ai_key ubuntu@<서버IP>:/home/ubuntu/remote_file.txt ./
```

**rsync (폴더 동기화, 더 효율적):**
```bash
# 프로젝트 폴더 전체 업로드
rsync -avz -e "ssh -i ~/.ssh/oci_ai_key" ./my-project/ ubuntu@<서버IP>:/home/ubuntu/my-project/

# -a: 권한/타임스탬프 보존
# -v: 상세 출력
# -z: 압축 전송
```

### 3.3 데이터베이스 클라이언트

**TablePlus 설치 (Homebrew):**
```bash
brew install --cask tableplus
```

**SSH 터널 연결 설정:**
1. TablePlus 실행 → New Connection → PostgreSQL
2. Connection 탭:
   - Host: `localhost`
   - Port: `5432`
   - Database: `ai_dev`
3. SSH 탭:
   - Host: `<서버 공인 IP>`
   - User: `ubuntu`
   - Key: `~/.ssh/oci_ai_key`

### 3.4 VS Code Remote-SSH

**확장 설치:**
- Remote - SSH (ms-vscode-remote.remote-ssh)

**사용법:**
1. `Cmd+Shift+P` → "Remote-SSH: Connect to Host"
2. `oci-ai` 선택 (~/.ssh/config에서 읽어옴)
3. 서버의 `/home/ubuntu` 폴더 열기
4. 터미널, 파일 탐색, Git 모두 서버에서 직접 실행

---

## 4. 서버 패키지 설치

### 4.1 기본 패키지

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget vim build-essential
```

### 4.2 Docker 설치

```bash
# Docker 공식 GPG 키 추가
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Docker 저장소 추가
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker 설치
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 현재 사용자를 docker 그룹에 추가 (sudo 없이 docker 명령 사용)
sudo usermod -aG docker $USER
newgrp docker  # 즉시 적용
```

**확인:**
```bash
docker --version
docker compose version
```

### 4.3 nginx (웹 서버)

```bash
sudo apt install -y nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

### 4.4 PostgreSQL + pgVector

**docker-compose.yml 작성:**

```yaml
version: '3.8'

services:
  postgres:
    image: ankane/pgvector:latest
    container_name: ai-postgres
    environment:
      POSTGRES_USER: aiuser
      POSTGRES_PASSWORD: securepassword
      POSTGRES_DB: ai_dev
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  pgdata:
```

**실행:**
```bash
docker compose up -d
```

**확인:**
```bash
docker ps
docker logs ai-postgres
```

---

## 5. Python 환경 설정

### 5.1 Python 가상환경 생성

**venv를 사용하는 이유:**
- 프로젝트마다 독립적인 패키지 버전 관리
- 시스템 Python 오염 방지
- `requirements.txt`로 의존성 재현 가능

```bash
# Python 3.10 설치
sudo apt install -y python3.10 python3.10-venv python3-pip

# 가상환경 생성
python3 -m venv ~/ai-venv

# 활성화
source ~/ai-venv/bin/activate

# 비활성화 (작업 완료 후)
deactivate
```

### 5.2 AI 패키지 설치

```bash
pip install --upgrade pip

# 기본 패키지
pip install numpy pandas matplotlib jupyter

# AI 프레임워크
pip install torch torchvision  # PyTorch
pip install transformers       # Hugging Face

# 데이터베이스
pip install psycopg2-binary    # PostgreSQL 드라이버
pip install pgvector           # pgVector Python 클라이언트

# 웹 프레임워크
pip install flask fastapi uvicorn

# LLM/Agent
pip install openai langchain langchain-community
```

### 5.3 requirements.txt 생성

```bash
pip freeze > requirements.txt
```

**다른 환경에서 복원:**
```bash
pip install -r requirements.txt
```

---

## 6. PostgreSQL + pgVector 설정

### 6.1 pgVector 확장 활성화

```bash
docker exec -it ai-postgres psql -U aiuser -d ai_dev
```

```sql
-- pgVector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 확인
\dx
```

### 6.2 벡터 테이블 생성 예시

```sql
-- 문서 임베딩 저장용 테이블
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI ada-002 차원
    created_at TIMESTAMP DEFAULT NOW()
);

-- 벡터 인덱스 생성 (유사도 검색 최적화)
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

### 6.3 Python에서 벡터 삽입/검색

```python
import psycopg2
from pgvector.psycopg2 import register_vector

# 연결 (SSH 터널 사용 시 localhost)
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="ai_dev",
    user="aiuser",
    password="securepassword"
)
register_vector(conn)

cur = conn.cursor()

# 벡터 삽입
embedding = [0.1] * 1536  # 실제로는 OpenAI API로 생성
cur.execute(
    "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
    ("샘플 문서", embedding)
)
conn.commit()

# 유사도 검색 (코사인 유사도)
query_embedding = [0.2] * 1536
cur.execute(
    """
    SELECT content, 1 - (embedding <=> %s) AS similarity
    FROM documents
    ORDER BY embedding <=> %s
    LIMIT 5
    """,
    (query_embedding, query_embedding)
)

for row in cur.fetchall():
    print(f"문서: {row[0]}, 유사도: {row[1]:.4f}")
```

### 6.4 pgVector 연산자

| 연산자 | 의미 | 사용 사례 |
|--------|------|-----------|
| `<->` | L2 거리 (유클리드) | 일반적인 거리 계산 |
| `<=>` | 코사인 거리 (1 - 코사인 유사도) | 텍스트 임베딩 검색 |
| `<#>` | 내적 (Inner Product) | 추천 시스템 |

**예시:**
```sql
-- 코사인 유사도 검색 (가장 일반적)
SELECT * FROM documents ORDER BY embedding <=> '[0.1,0.2,...]' LIMIT 10;

-- L2 거리 검색
SELECT * FROM documents ORDER BY embedding <-> '[0.1,0.2,...]' LIMIT 10;
```

---

## 환경 구축 완료

이제 다음 작업을 수행할 수 있다:

1. **로컬 개발**: VS Code Remote-SSH로 서버에서 직접 코드 작성
2. **데이터베이스 관리**: TablePlus로 PostgreSQL 시각화
3. **벡터 검색**: pgVector로 문서 유사도 검색 구현
4. **컨테이너 관리**: Docker Compose로 서비스 배포

**다음 단계**: LangChain을 활용한 RAG 시스템 구축
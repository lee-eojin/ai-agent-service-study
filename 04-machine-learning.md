# 머신러닝 (Machine Learning)

## 학습 목표

이 문서에서는 PyTorch를 활용한 머신러닝 기본 개념과 실습을 다룬다.

1. PyTorch 기초: 텐서 연산, 자동 미분, 주요 모듈 이해
2. 선형 회귀: 경사하강법을 통한 가중치 최적화 원리 학습
3. 로지스틱 회귀: 이진 분류 문제 해결
4. 신경망 구조: ANN, RNN, LSTM 구현 및 응용

---

## 1. PyTorch 기초

### 1.1 PyTorch 소개

PyTorch는 Facebook AI Research(FAIR)에서 개발한 딥러닝 프레임워크로, NumPy와 유사한 텐서 연산과 자동 미분 기능을 제공한다.

주요 모듈:
- `torch`: 텐서 연산 및 수학 함수
- `torch.autograd`: 자동 미분 (역전파)
- `torch.nn`: 신경망 레이어, 활성화 함수, 손실 함수
- `torch.optim`: 최적화 알고리즘 (SGD, Adam 등)

설치:
```bash
# CPU 버전
pip install torch torchvision

# CUDA 11.8 (NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> PyTorch vs TensorFlow
>
> PyTorch 장점:
> - 직관적인 Python 문법 (동적 그래프)
> - 디버깅이 쉬움 (일반 Python 디버거 사용 가능)
> - 연구 및 프로토타이핑에 유리
>
> TensorFlow 장점:
> - 프로덕션 배포에 강함 (TensorFlow Serving, TFLite)
> - Google 생태계와의 통합 (TPU, Colab)
>
> 최근 트렌드:
> - 학계: PyTorch 압도적 우세
> - 산업계: 두 프레임워크 모두 활발히 사용

### 1.2 텐서 (Tensor)

텐서의 차원:
- Scalar (0D): 단일 숫자 (예: 온도 25°C)
- Vector (1D): 숫자 배열 (예: [1, 2, 3])
- Matrix (2D): 2차원 배열 (예: 이미지의 한 채널)
- Tensor (3D+): 3차원 이상 (예: RGB 이미지는 3D 텐서)

```python
import torch
import numpy as np

# 텐서 생성
t1 = torch.FloatTensor([1, 2, 3, 4])
t2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# NumPy 배열에서 변환
arr = np.array([1, 2, 3])
t3 = torch.from_numpy(arr)

# 기본 속성
print(t2.shape)       # torch.Size([2, 2])
print(t2.dtype)       # torch.float32
print(t2.device)      # cpu (또는 cuda:0)
```

### 1.3 텐서 연산

기본 연산:
```python
import torch

# 산술 연산
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)          # tensor([5., 7., 9.])
print(a * b)          # 원소별 곱셈
print(torch.dot(a, b))  # 내적: 32.0

# 브로드캐스팅 (자동 크기 맞춤)
m1 = torch.FloatTensor([[1, 2]])  # (1, 2)
m2 = torch.FloatTensor([[3], [4]])  # (2, 1)
print(m1 + m2)
# tensor([[4., 5.],
#         [5., 6.]])
```

행렬 곱셈:
```python
m1 = torch.FloatTensor([[1, 2], [3, 4]])  # (2, 2)
m2 = torch.FloatTensor([[1], [2]])         # (2, 1)

# 행렬곱 (Matrix Multiplication)
print(torch.matmul(m1, m2))  # (2, 1)
# tensor([[ 5.],
#         [11.]])

# @ 연산자 (Python 3.5+)
print(m1 @ m2)  # 동일
```

> 원소별 곱 vs 행렬곱
>
> ```python
> a = torch.tensor([[1, 2], [3, 4]])
> b = torch.tensor([[2, 0], [1, 2]])
>
> # 원소별 곱 (Element-wise)
> print(a * b)
> # tensor([[2, 0],
> #         [3, 8]])
>
> # 행렬곱 (Matrix Multiplication)
> print(a @ b)
> # tensor([[ 4,  4],
> #         [10,  8]])
> ```
>
> 주의: 신경망의 가중치 연산은 대부분 행렬곱(`@` 또는 `matmul`)이다.

통계 함수:
```python
t = torch.FloatTensor([[1, 2], [3, 4]])

print(t.mean())        # 2.5 (전체 평균)
print(t.mean(dim=0))   # tensor([2., 3.]) (열 기준)
print(t.mean(dim=1))   # tensor([1.5, 3.5]) (행 기준)

print(t.sum())         # 10.0
print(t.max())         # 4.0
print(t.argmax())      # 3 (최댓값의 인덱스)
```

형태 변환:
```python
t = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])  # (3, 2)

# view: 형태 변경 (메모리 공유)
print(t.view(2, 3))    # (2, 3)
print(t.view(-1, 1))   # (6, 1) - -1은 자동 계산

# squeeze/unsqueeze: 차원 제거/추가
t2 = torch.FloatTensor([1, 2, 3])  # (3,)
print(t2.unsqueeze(0))  # (1, 3) - 행 벡터
print(t2.unsqueeze(1))  # (3, 1) - 열 벡터

t3 = torch.FloatTensor([[1], [2]])  # (2, 1)
print(t3.squeeze())     # tensor([1., 2.]) - (2,)
```

---

## 2. 선형 회귀 (Linear Regression)

### 2.1 선형 회귀 개념

선형 회귀는 입력 데이터(x)와 출력 데이터(y) 사이의 선형 관계를 학습하는 알고리즘이다.

가설 (Hypothesis):
```
H(x) = Wx + b
```

비용 함수 (Cost Function):
```
MSE = (1/n) Σ (y - H(x))²
```

최적화:
- 경사하강법(Gradient Descent)으로 W와 b를 업데이트
- 학습률(learning rate)로 업데이트 크기 조절

> 학습률(Learning Rate) 설정 팁
>
> 학습률이 너무 크면:
> - 손실이 발산하거나 진동
> - 최적값을 지나쳐버림
>
> 학습률이 너무 작으면:
> - 학습이 매우 느림
> - 지역 최솟값(local minimum)에 빠질 위험
>
> 일반적인 시작값:
> - 선형 회귀: 0.01 ~ 0.1
> - 신경망: 0.001 ~ 0.01
> - Adam 옵티마이저: 0.001 (기본값)
>
> 실무 팁:
> - 학습률 스케줄러 사용 (점진적 감소)
> - 처음엔 큰 값으로 시작해서 점점 줄이기

### 2.2 수동 구현 (저수준)

```python
import torch
import torch.optim as optim

# 훈련 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 가중치 초기화 (requires_grad=True로 자동 미분 활성화)
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 옵티마이저 설정
optimizer = optim.SGD([W, b], lr=0.01)

# 학습 루프
nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # 가설 (예측)
    hypothesis = x_train * W + b

    # 비용 계산 (MSE)
    cost = torch.mean((hypothesis - y_train) ** 2)

    # 경사하강법
    optimizer.zero_grad()  # 기울기 초기화
    cost.backward()        # 역전파 (자동 미분)
    optimizer.step()       # 가중치 업데이트

    # 로그 출력
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}/{nb_epochs} | Cost: {cost.item():.6f} | W: {W.item():.3f}, b: {b.item():.3f}")
```

출력 예시:
```
Epoch    0/1000 | Cost: 18.666666 | W: 0.187, b: 0.080
Epoch  100/1000 | Cost: 0.000011 | W: 1.999, b: 0.004
Epoch  200/1000 | Cost: 0.000000 | W: 2.000, b: 0.000
...
```

### 2.3 nn.Module 활용 (고수준)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 모델 정의
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 입력 1개, 출력 1개

    def forward(self, x):
        return self.linear(x)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델 생성
model = LinearRegressionModel()

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습
for epoch in range(1000):
    # 예측
    prediction = model(x_train)

    # 손실 계산
    loss = criterion(prediction, y_train)

    # 역전파 및 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# 학습된 가중치 확인
print(f"\nW: {model.linear.weight.item():.3f}")
print(f"b: {model.linear.bias.item():.3f}")
```

### 2.4 다변량 선형 회귀

입력 변수가 여러 개인 경우:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 (x1, x2, x3) → y
x_train = torch.FloatTensor([
    [73, 80, 75],
    [93, 88, 93],
    [89, 91, 90],
    [96, 98, 100],
    [73, 66, 70]
])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델
model = nn.Linear(3, 1)

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-5)

# 학습
for epoch in range(2000):
    prediction = model(x_train)
    loss = criterion(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.2f}")

# 새 데이터 예측
new_data = torch.FloatTensor([[80, 85, 90]])
pred = model(new_data)
print(f"\n예측값: {pred.item():.2f}")
```

---

## 3. 로지스틱 회귀 (Logistic Regression)

### 3.1 로지스틱 회귀 개념

이진 분류를 위한 알고리즘. 선형 회귀의 출력에 시그모이드 함수를 적용하여 0~1 사이의 확률로 변환한다.

시그모이드 함수:
```
σ(z) = 1 / (1 + e^(-z))
```

가설:
```
H(x) = σ(Wx + b)
```

손실 함수:
- Binary Cross-Entropy Loss
```
BCE = -[y·log(H(x)) + (1-y)·log(1-H(x))]
```

### 3.2 로지스틱 회귀 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 데이터 (x1, x2) → y (0 또는 1)
x_train = torch.FloatTensor([
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 3],
    [5, 3],
    [6, 2]
])
y_train = torch.FloatTensor([[0], [0], [0], [1], [1], [1]])

# 모델
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()

# 손실 함수와 옵티마이저
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습
for epoch in range(1000):
    prediction = model(x_train)
    loss = criterion(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

# 예측
with torch.no_grad():
    test_data = torch.FloatTensor([[2.5, 2.0]])
    pred = model(test_data)
    print(f"\n예측 확률: {pred.item():.4f}")
    print(f"클래스: {1 if pred >= 0.5 else 0}")
```

> 손실 함수 선택 가이드
>
> | 문제 유형 | 손실 함수 | PyTorch 함수 |
> |-----------|-----------|--------------|
> | 회귀 (연속값 예측) | MSE, MAE | `nn.MSELoss()`, `nn.L1Loss()` |
> | 이진 분류 | Binary Cross-Entropy | `nn.BCELoss()` |
> | 다중 클래스 분류 | Cross-Entropy | `nn.CrossEntropyLoss()` |
>
> 주의:
> - `BCELoss`는 출력이 0~1 사이여야 함 (시그모이드 적용 필요)
> - `CrossEntropyLoss`는 내부에 Softmax 포함 (따로 적용하면 안 됨)

---

## 4. 인공신경망 (Artificial Neural Network)

### 4.1 퍼셉트론 (Perceptron)

단층 퍼셉트론:
- AND, OR 게이트는 구현 가능
- XOR 게이트는 불가능 (선형 분리 불가)

다층 퍼셉트론 (MLP):
- 은닉층(Hidden Layer)을 추가하여 비선형 문제 해결
- 활성화 함수로 비선형성 추가

### 4.2 활성화 함수

주요 활성화 함수:
```python
import torch
import torch.nn as nn

x = torch.linspace(-5, 5, 100)

# ReLU (Rectified Linear Unit)
relu = nn.ReLU()
y_relu = relu(x)
# f(x) = max(0, x)

# Sigmoid
sigmoid = nn.Sigmoid()
y_sigmoid = sigmoid(x)
# f(x) = 1 / (1 + e^(-x))

# Tanh
tanh = nn.Tanh()
y_tanh = tanh(x)
# f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

# Leaky ReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
y_leaky = leaky_relu(x)
# f(x) = x if x > 0 else 0.01 * x
```

> 활성화 함수 선택 가이드
>
> ReLU (가장 일반적):
> - 장점: 계산 빠름, 기울기 소실 문제 완화
> - 단점: Dying ReLU (음수에서 기울기 0)
> - 사용: 은닉층 대부분
>
> Leaky ReLU:
> - ReLU의 Dying 문제 해결
> - 음수에서도 작은 기울기 유지
>
> Sigmoid:
> - 출력: 0~1 (확률 해석 가능)
> - 사용: 이진 분류 출력층
> - 단점: 기울기 소실 문제
>
> Tanh:
> - 출력: -1~1
> - Sigmoid보다 중심이 0에 가까워 학습 안정
> - 단점: 기울기 소실 문제
>
> Softmax:
> - 다중 클래스 분류 출력층
> - 모든 클래스 확률의 합 = 1

### 4.3 MNIST 손글씨 분류

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 데이터 로딩
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 모델 정의 (28x28 = 784 입력)
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10개 클래스
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)  # 평탄화
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Softmax는 CrossEntropyLoss에 포함
        return x

model = MNISTClassifier()

# 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
nb_epochs = 5
for epoch in range(nb_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{nb_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

# 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f"\n테스트 정확도: {accuracy:.2f}%")
```

> 과적합(Overfitting) 방지 기법
>
> 1. Dropout:
> ```python
> class Model(nn.Module):
>     def __init__(self):
>         super().__init__()
>         self.fc1 = nn.Linear(784, 128)
>         self.dropout = nn.Dropout(0.5)  # 50% 뉴런 비활성화
>         self.fc2 = nn.Linear(128, 10)
>
>     def forward(self, x):
>         x = F.relu(self.fc1(x))
>         x = self.dropout(x)  # 학습 시에만 적용
>         return self.fc2(x)
> ```
>
> 2. Batch Normalization:
> - 각 레이어의 입력을 정규화
> - 학습 안정화 및 속도 향상
>
> 3. Early Stopping:
> - 검증 손실이 증가하면 학습 중단
>
> 4. Data Augmentation:
> - 이미지: 회전, 크롭, 플립 등
> - 텍스트: 동의어 치환, 역번역 등

---

## 5. 순환신경망 (RNN)

### 5.1 RNN 개념

RNN (Recurrent Neural Network)은 시퀀스 데이터를 처리하는 신경망으로, 이전 단계의 은닉 상태를 현재 단계의 입력으로 사용한다.

사용 사례:
- 자연어 처리 (텍스트 분류, 기계 번역)
- 시계열 예측 (주가, 날씨)
- 음성 인식

RNN 구조:
```
h_t = tanh(W_hh * h_(t-1) + W_xh * x_t + b)
```

### 5.2 RNN 구현

```python
import torch
import torch.nn as nn

# RNN 레이어
rnn = nn.RNN(
    input_size=10,   # 입력 특성 수
    hidden_size=20,  # 은닉 상태 크기
    num_layers=2,    # RNN 레이어 수
    batch_first=True # (batch, seq, feature) 순서
)

# 입력: (batch_size=3, seq_len=5, input_size=10)
input_data = torch.randn(3, 5, 10)

# 초기 은닉 상태: (num_layers=2, batch_size=3, hidden_size=20)
h0 = torch.zeros(2, 3, 20)

# 순전파
output, hn = rnn(input_data, h0)

print(output.shape)  # (3, 5, 20) - 모든 시간 단계의 출력
print(hn.shape)      # (2, 3, 20) - 마지막 은닉 상태
```

간단한 문자 예측 모델:
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터: "hello" → "ello"
char_set = ['h', 'e', 'l', 'o']
char_to_idx = {c: i for i, c in enumerate(char_set)}
idx_to_char = {i: c for i, c in enumerate(char_set)}

# 입력/출력
input_seq = [char_to_idx[c] for c in "hell"]
target_seq = [char_to_idx[c] for c in "ello"]

# 원-핫 인코딩
input_one_hot = [torch.eye(4)[i] for i in input_seq]
input_tensor = torch.stack(input_one_hot).unsqueeze(0)  # (1, 4, 4)
target_tensor = torch.LongTensor(target_seq)

# 모델
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out.view(-1, out.size(2))

model = CharRNN(input_size=4, hidden_size=8, output_size=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습
for epoch in range(100):
    output = model(input_tensor)
    loss = criterion(output, target_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# 예측
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    predicted_chars = ''.join([idx_to_char[i.item()] for i in predicted])
    print(f"\n입력: hell → 예측: {predicted_chars}")
```

### 5.3 LSTM (Long Short-Term Memory)

LSTM의 장점:
- RNN의 기울기 소실 문제 해결
- 장기 의존성 학습 가능 (긴 시퀀스 처리)

LSTM 게이트:
1. Forget Gate: 이전 정보를 얼마나 잊을지 결정
2. Input Gate: 새 정보를 얼마나 받아들일지 결정
3. Output Gate: 현재 상태를 얼마나 출력할지 결정

```python
import torch
import torch.nn as nn

# LSTM 레이어
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

# 입력
input_data = torch.randn(3, 5, 10)

# 초기 상태 (은닉 상태 + 셀 상태)
h0 = torch.zeros(2, 3, 20)
c0 = torch.zeros(2, 3, 20)

# 순전파
output, (hn, cn) = lstm(input_data, (h0, c0))

print(output.shape)  # (3, 5, 20)
print(hn.shape)      # (2, 3, 20)
print(cn.shape)      # (2, 3, 20)
```

LSTM 문자 예측:
```python
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.view(-1, out.size(2))

# 사용법은 CharRNN과 동일
model = CharLSTM(input_size=4, hidden_size=8, output_size=4)
```

> RNN vs LSTM vs GRU
>
> | 모델 | 장점 | 단점 | 사용 사례 |
> |------|------|------|-----------|
> | RNN | 간단, 빠름 | 기울기 소실, 장기 의존성 약함 | 짧은 시퀀스 |
> | LSTM | 장기 의존성 학습 가능 | 파라미터 많음, 느림 | 긴 시퀀스, 복잡한 패턴 |
> | GRU | LSTM보다 빠름, 성능 유사 | LSTM보다 약간 낮은 표현력 | 일반적인 시퀀스 처리 |
>
> 실무 팁:
> - 먼저 LSTM으로 시작
> - 속도가 중요하면 GRU 고려
> - 최근에는 Transformer가 RNN 계열을 대체하는 추세 (병렬 처리 가능)

---

## 6. 실습 프로젝트

### 6.1 배추 가격 예측 (선형 회귀)

데이터 형식 (cabbage.csv):
```
평균기온(°C),강수량(mm),가격(원/kg)
5.2,10.5,3200
7.8,25.3,2800
12.3,5.1,2500
...
```

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 로드
df = pd.read_csv("cabbage.csv", encoding="utf-8")
X = df[['평균기온(°C)', '강수량(mm)']].values
y = df['가격(원/kg)'].values.reshape(-1, 1)

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 텐서 변환
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# 모델
model = nn.Linear(2, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습
for epoch in range(2000):
    prediction = model(X_train)
    loss = criterion(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.2f}")

# 평가
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = criterion(test_pred, y_test)
    print(f"\n테스트 손실: {test_loss.item():.2f}")

# 새 데이터 예측
new_data = torch.FloatTensor([[10.0, 15.0]])  # 10°C, 15mm
pred = model(new_data)
print(f"예측 가격: {pred.item():.0f}원/kg")
```

### 6.2 Flask 서버로 배포

```python
from flask import Flask, request, jsonify
import torch
import torch.nn as nn

app = Flask(__name__)

# 학습된 모델 로드
model = nn.Linear(2, 1)
model.load_state_dict(torch.load("cabbage_model.pth"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    temp = data["temperature"]
    rainfall = data["rainfall"]

    # 예측
    input_tensor = torch.FloatTensor([[temp, rainfall]])
    with torch.no_grad():
        prediction = model(input_tensor)

    return jsonify({
        "temperature": temp,
        "rainfall": rainfall,
        "predicted_price": round(prediction.item(), 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

사용법:
```bash
# 모델 저장
torch.save(model.state_dict(), "cabbage_model.pth")

# 서버 실행
python app.py

# API 호출
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 10, "rainfall": 15}'
```

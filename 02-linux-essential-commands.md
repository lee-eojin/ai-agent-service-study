# 리눅스 필수 명령어

## 개요

리눅스는 다중 사용자와 다중 작업을 지원하는 운영체제로, 기본적인 구조와 명령 체계가 macOS와 크게 다르지 않다. 시스템 관리와 사용자 작업 모두 명령어 기반으로 이뤄지며, 이를 통해 서버 운영과 개발 환경 구성이 가능하다.

---

## 1. 시스템 정보 및 관리

### 1.1 시스템 정보 확인

```bash
# 커널 정보 및 시스템 아키텍처
uname -a

# 배포판 버전 확인 (Ubuntu, Debian 등)
lsb_release -a

# macOS 버전 확인
sw_vers
```

출력 예시:
```
Linux ai-dev-server 5.15.0-1023-oracle #29-Ubuntu SMP x86_64 GNU/Linux
```

### 1.2 시스템 종료 및 재부팅

```bash
# 즉시 종료
sudo shutdown -h now

# 10분 후 종료 (예약)
sudo shutdown -h +10

# 재부팅
sudo reboot

# 취소
sudo shutdown -c
```

### 1.3 서비스 관리

systemctl (Linux):
```bash
# 서비스 시작
sudo systemctl start nginx

# 서비스 중지
sudo systemctl stop nginx

# 서비스 재시작
sudo systemctl restart nginx

# 부팅 시 자동 시작 설정
sudo systemctl enable nginx

# 서비스 상태 확인
sudo systemctl status nginx

# 전체 서비스 목록
systemctl list-units --type=service
```

launchctl (macOS):
```bash
# 서비스 시작
launchctl start com.example.service

# 서비스 중지
launchctl stop com.example.service

# 서비스 목록
launchctl list
```

---

## 2. 디스크 및 파일 시스템

### 2.1 디스크 사용량 확인

```bash
# 파일 시스템별 디스크 사용량 (human-readable)
df -h

# 특정 디렉토리 크기
du -sh /var/log

# 하위 디렉토리별 크기 (상위 10개)
du -h /home/ubuntu | sort -rh | head -n 10
```

출력 예시:
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        40G   12G   26G  32% /
tmpfs           2.0G     0  2.0G   0% /dev/shm
```

### 2.2 마운트 관리

```bash
# 현재 마운트 상태 확인
mount

# 새 디스크 마운트
sudo mount /dev/sdb1 /mnt/data

# 언마운트
sudo umount /mnt/data

# /etc/fstab에 등록 (부팅 시 자동 마운트)
echo "/dev/sdb1 /mnt/data ext4 defaults 0 2" | sudo tee -a /etc/fstab
```

---

## 3. 사용자 및 권한 관리

### 3.1 사용자 계정

```bash
# 사용자 추가 (기본)
sudo useradd newuser

# 사용자 추가 (홈 디렉토리 생성, 대화형)
sudo adduser newuser

# 비밀번호 설정
sudo passwd newuser

# 사용자 삭제 (홈 디렉토리 포함)
sudo userdel -r newuser

# 현재 로그인 사용자 확인
whoami

# 전체 사용자 목록
cat /etc/passwd
```

### 3.2 그룹 관리

```bash
# 그룹 생성
sudo groupadd developers

# 사용자를 그룹에 추가
sudo usermod -aG developers newuser

# 사용자 그룹 확인
groups newuser

# 사용자의 기본 그룹 변경
sudo usermod -g developers newuser
```

### 3.3 sudo 권한 부여

```bash
# sudoers 파일 편집 (안전한 방법)
sudo visudo

# 특정 사용자에게 sudo 권한 부여 (파일에 추가)
newuser ALL=(ALL:ALL) ALL

# 또는 sudo 그룹에 추가 (Ubuntu 기본 설정)
sudo usermod -aG sudo newuser
```

---

## 4. 파일 및 디렉토리 관리

### 4.1 기본 탐색

```bash
# 현재 경로 확인
pwd

# 디렉토리 이동
cd /var/log
cd ~          # 홈 디렉토리
cd -          # 이전 경로

# 파일 목록 (상세, 숨김 파일 포함)
ls -al

# 파일 개수 세기
ls -1 | wc -l
```

### 4.2 파일 생성 및 조작

```bash
# 빈 파일 생성 (또는 타임스탬프 갱신)
touch newfile.txt

# 디렉토리 생성 (중첩 경로도 한 번에)
mkdir -p project/src/components

# 파일 복사
cp source.txt dest.txt
cp -r /source_dir /dest_dir  # 디렉토리 재귀 복사

# 파일 이동/이름 변경
mv old.txt new.txt
mv file.txt /tmp/

# 파일 삭제
rm file.txt
rm -rf directory/  # 디렉토리 강제 삭제 (주의!)
```

### 4.3 파일 내용 확인

```bash
# 전체 내용 출력
cat file.txt

# 앞 10줄
head -n 10 file.txt

# 뒤 20줄
tail -n 20 file.txt

# 실시간 로그 추적 (Ctrl+C로 종료)
tail -f /var/log/syslog

# 페이지 단위 보기 (스크롤 가능)
less file.txt   # q로 종료
more file.txt
```

### 4.4 파일 검색

```bash
# 파일 이름으로 찾기
find /home/ubuntu -name "*.log"

# 특정 크기 이상 파일 찾기 (100MB 이상)
find /var -size +100M

# 7일 이내 수정된 파일
find /home/ubuntu -mtime -7

# 디렉토리만 찾기
find /etc -type d

# 실행 권한 있는 파일
find . -perm /a+x
```

---

## 5. 권한 관리

### 5.1 chmod (권한 변경)

```bash
# 8진수 표기법
chmod 755 script.sh   # rwxr-xr-x (소유자: 읽기+쓰기+실행, 그룹/기타: 읽기+실행)
chmod 644 file.txt    # rw-r--r-- (소유자: 읽기+쓰기, 그룹/기타: 읽기)

# 심볼릭 표기법
chmod +x script.sh           # 모든 사용자에게 실행 권한 추가
chmod u+w,g-w,o-w file.txt   # 소유자만 쓰기 가능
chmod -R 755 /var/www        # 디렉토리 재귀 적용
```

권한 번호 이해:
- 4 = 읽기 (r)
- 2 = 쓰기 (w)
- 1 = 실행 (x)
- 합산: 7(4+2+1) = rwx, 6(4+2) = rw-, 5(4+1) = r-x

### 5.2 chown (소유자 변경)

```bash
# 소유자 변경
sudo chown ubuntu file.txt

# 소유자와 그룹 동시 변경
sudo chown ubuntu:developers file.txt

# 디렉토리 재귀 적용
sudo chown -R ubuntu:ubuntu /home/ubuntu/project
```

---

## 6. 텍스트 검색 및 처리

### 6.1 grep (패턴 검색)

```bash
# 기본 검색
grep "error" /var/log/syslog

# 대소문자 무시
grep -i "error" log.txt

# 재귀 검색 (디렉토리 전체)
grep -r "TODO" /home/ubuntu/project

# 줄 번호 표시
grep -n "function" script.js

# 정규표현식 (-E 옵션)
grep -E "^[0-9]{3}" file.txt   # 숫자 3자리로 시작하는 줄

# 파일 이름만 출력
grep -l "import" *.py

# 매칭되지 않는 줄 (반대)
grep -v "DEBUG" log.txt
```

### 6.2 파이프와 조합

```bash
# 프로세스에서 특정 단어 검색
ps aux | grep nginx

# 로그에서 에러만 추출해 카운트
cat /var/log/app.log | grep "ERROR" | wc -l

# 특정 확장자 파일만 찾아 검색
find . -name "*.js" | xargs grep "console.log"
```

### 6.3 sed (텍스트 치환)

```bash
# 문자열 치환 (첫 번째 매칭만)
sed 's/old/new/' file.txt

# 모든 매칭 치환 (g 옵션)
sed 's/old/new/g' file.txt

# 파일 직접 수정 (-i 옵션)
sed -i 's/localhost/127.0.0.1/g' config.conf

# 특정 줄 삭제
sed '5d' file.txt            # 5번째 줄 삭제
sed '/pattern/d' file.txt    # 패턴 매칭 줄 삭제
```

### 6.4 awk (텍스트 처리)

```bash
# 특정 컬럼만 출력 (공백 구분)
awk '{print $1, $3}' file.txt

# 조건 필터링
awk '$3 > 100 {print $0}' data.txt   # 3번째 컬럼이 100 초과인 줄

# CSV 처리 (구분자 지정)
awk -F',' '{print $2}' data.csv

# 합계 계산
awk '{sum += $1} END {print sum}' numbers.txt
```

---

## 7. 프로세스 관리

### 7.1 프로세스 확인

```bash
# 전체 프로세스 (표준 포맷)
ps -ef

# 사용자별 프로세스 (BSD 스타일)
ps aux

# 트리 구조로 보기
ps auxf
pstree

# 실시간 모니터링
top      # 기본 모니터
htop     # 향상된 UI (설치 필요: sudo apt install htop)

# 특정 프로세스만 찾기
ps aux | grep nginx
```

### 7.2 프로세스 종료

```bash
# PID로 종료 (정상 종료 시도)
kill 1234

# 강제 종료
kill -9 1234
kill -SIGKILL 1234  # 동일

# 이름으로 종료
pkill nginx

# 모든 매칭 프로세스 종료
killall python3
```

### 7.3 백그라운드 실행

```bash
# 백그라운드로 실행
./long_task.sh &

# 중단된 작업 목록 확인
jobs

# 포그라운드로 전환
fg %1   # 작업 번호 1

# 백그라운드로 전환
bg %1

# 로그아웃 후에도 유지 (nohup)
nohup python3 server.py &

# 출력 리다이렉션 (로그 파일로)
nohup ./script.sh > output.log 2>&1 &
```

---

## 8. 네트워크

### 8.1 네트워크 인터페이스

```bash
# 네트워크 설정 확인 (구식)
ifconfig

# 네트워크 설정 확인 (최신)
ip addr show
ip a   # 축약형

# 특정 인터페이스만 보기
ip addr show eth0

# 라우팅 테이블
ip route
route -n
```

### 8.2 연결 테스트

```bash
# 핑 테스트 (연결 확인)
ping google.com
ping -c 4 8.8.8.8   # 4번만 전송

# 경로 추적
traceroute google.com

# DNS 조회
nslookup google.com
dig google.com

# 포트 통신 테스트
telnet 192.168.1.100 22
nc -zv 192.168.1.100 22   # netcat (더 범용적)
```

### 8.3 네트워크 상태

```bash
# 열린 포트 및 연결 확인
netstat -tulnp   # t=TCP, u=UDP, l=LISTEN, n=숫자 포트, p=프로세스

# 특정 포트 사용 프로세스 확인
lsof -i :80
lsof -i :5432

# 모든 네트워크 연결
ss -tuln   # netstat의 최신 대체 명령
```

### 8.4 파일 전송

```bash
# SSH 파일 복사 (로컬 → 원격)
scp file.txt user@host:/path/to/destination

# 원격 → 로컬
scp user@host:/path/to/file.txt ./

# 디렉토리 복사 (-r 옵션)
scp -r project/ user@host:/home/user/

# rsync (더 효율적, 증분 전송)
rsync -avz local_dir/ user@host:/remote_dir/
# -a: 권한 보존, -v: 상세 출력, -z: 압축
```

---

## 9. 시스템 로그

### 9.1 로그 위치

```bash
# 주요 로그 디렉토리
/var/log/

# 시스템 전체 로그
/var/log/syslog      # Ubuntu/Debian
/var/log/messages    # CentOS/RHEL

# 인증 로그
/var/log/auth.log

# 커널 로그
dmesg
journalctl -k
```

### 9.2 journalctl (systemd 로그)

```bash
# 전체 로그
journalctl

# 특정 서비스 로그
journalctl -u nginx

# 실시간 로그 추적
journalctl -f
journalctl -u docker -f

# 최근 50줄
journalctl -n 50

# 오늘 로그만
journalctl --since today

# 특정 시간 범위
journalctl --since "2025-01-01 00:00:00" --until "2025-01-02 00:00:00"
```

---

## 10. 셸 프로그래밍

### 10.1 기본 스크립트

```bash
#!/bin/bash
# backup.sh - 간단한 백업 스크립트

# 변수 선언
BACKUP_DIR="/backup"
SOURCE_DIR="/home/ubuntu/project"
DATE=$(date +%Y%m%d)

# 디렉토리 생성
mkdir -p $BACKUP_DIR

# 압축 백업
tar -czf $BACKUP_DIR/backup_$DATE.tar.gz $SOURCE_DIR

echo "백업 완료: backup_$DATE.tar.gz"
```

실행 권한 부여:
```bash
chmod +x backup.sh
./backup.sh
```

### 10.2 조건문

```bash
#!/bin/bash

FILE="/etc/nginx/nginx.conf"

if [ -f "$FILE" ]; then
    echo "파일이 존재합니다"
elif [ -d "$FILE" ]; then
    echo "디렉토리입니다"
else
    echo "파일이 없습니다"
fi
```

비교 연산자:
- `-f`: 파일 존재
- `-d`: 디렉토리 존재
- `-z`: 문자열이 비어있음
- `-n`: 문자열이 비어있지 않음
- `==`: 문자열 같음
- `-eq`: 숫자 같음
- `-gt`: 크다, `-lt`: 작다

### 10.3 반복문

```bash
# for 루프
for i in {1..5}; do
    echo "카운트: $i"
done

# 파일 순회
for file in /var/log/*.log; do
    echo "처리 중: $file"
done

# while 루프
count=0
while [ $count -lt 10 ]; do
    echo $count
    ((count++))
done
```

### 10.4 환경 설정

~/.bashrc (또는 ~/.zshrc):
```bash
# 환경 변수 설정
export PATH="$HOME/bin:$PATH"
export EDITOR="vim"

# alias 설정
alias ll='ls -alh'
alias gs='git status'
alias dc='docker compose'

# 함수 정의
function mkcd() {
    mkdir -p "$1" && cd "$1"
}

# 프롬프트 커스터마이징
PS1='\u@\h:\w\$ '
```

적용:
```bash
source ~/.bashrc
```

---

## 11. Docker 명령어

### 11.1 컨테이너 관리

```bash
# 이미지 실행 (포트 매핑, 백그라운드)
docker run -d -p 8080:80 --name webserver nginx

# 실행 중인 컨테이너 확인
docker ps

# 종료된 것 포함
docker ps -a

# 컨테이너 중지
docker stop webserver

# 컨테이너 시작
docker start webserver

# 컨테이너 삭제
docker rm webserver
docker rm -f webserver  # 강제 삭제
```

### 11.2 컨테이너 내부 접근

```bash
# Bash 셸 진입
docker exec -it webserver bash

# 단일 명령 실행
docker exec webserver ls /etc

# 로그 확인
docker logs webserver
docker logs -f webserver  # 실시간 추적
```

### 11.3 이미지 관리

```bash
# 이미지 목록
docker images

# 이미지 다운로드
docker pull ubuntu:22.04

# 이미지 삭제
docker rmi nginx

# 사용하지 않는 이미지 모두 삭제
docker image prune -a

# 빌드
docker build -t myapp:latest .
```

### 11.4 Docker Compose

```bash
# 서비스 시작 (백그라운드)
docker compose up -d

# 로그 확인
docker compose logs -f

# 서비스 중지 (컨테이너 유지)
docker compose stop

# 서비스 중지 및 삭제 (볼륨은 유지)
docker compose down

# 볼륨까지 삭제
docker compose down -v

# 특정 서비스만 재시작
docker compose restart postgres
```

---

## 12. 패키지 관리

### 12.1 APT (Ubuntu/Debian)

```bash
# 패키지 목록 업데이트
sudo apt update

# 설치된 패키지 업그레이드
sudo apt upgrade -y

# 패키지 검색
apt search nginx

# 패키지 설치
sudo apt install nginx

# 패키지 제거
sudo apt remove nginx

# 설정 파일까지 제거
sudo apt purge nginx

# 불필요한 패키지 정리
sudo apt autoremove
```

### 12.2 Homebrew (macOS)

```bash
# 패키지 설치
brew install wget

# GUI 앱 설치
brew install --cask visual-studio-code

# 업데이트
brew update
brew upgrade

# 패키지 검색
brew search python

# 설치된 패키지 목록
brew list
```

---

## 13. 유용한 조합 명령어

```bash
# 디스크 사용량 상위 10개 디렉토리
du -h /home | sort -rh | head -n 10

# 특정 포트 사용 중인 프로세스 종료
lsof -ti:8080 | xargs kill -9

# 로그에서 IP 주소 추출 및 카운트
cat /var/log/nginx/access.log | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | sort | uniq -c | sort -rn

# 가장 많이 사용하는 명령어 상위 10개
history | awk '{print $2}' | sort | uniq -c | sort -rn | head -n 10

# 특정 확장자 파일만 압축
find . -name "*.log" -type f | tar -czf logs.tar.gz -T -

# 메모리 사용량 상위 5개 프로세스
ps aux --sort=-%mem | head -n 6

# 오래된 파일 자동 삭제 (30일 이상)
find /tmp -type f -mtime +30 -delete
```

---

## 요약

이 문서에서 다룬 명령어들은 서버 운영, 개발 환경 구성, 트러블슈팅의 기본이 된다. 자주 사용하는 명령어는 alias로 등록하거나 셸 스크립트로 자동화하면 효율적이다. macOS와 리눅스는 대부분의 명령어를 공유하지만, 서비스 관리(systemctl vs launchctl)와 패키지 관리(apt vs brew) 부분에서 차이가 있다.
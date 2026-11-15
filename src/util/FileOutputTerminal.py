import os
import sys  # FileOutputTerminal 클래스를 위한 import

class FileOutputTerminal:
    """
    A class to duplicate terminal output to a file.

    This class captures everything printed to the terminal and also writes it to a specified log file.

    Attributes:
    -----------
    terminal : file-like object
        Standard output stream.
    log : file-like object
        File object where the output is logged.

    Methods:
    --------
    write(message):
        Writes the message to both the terminal and the log file.
    flush():
        Flushes the terminal and the log file.
    close():
        Closes the log file.
    open():
        Opens the log file.
    save_log():
        Closes and reopens the log file to save the current state.
    """  # docstring

    def __init__(self, filename):
        self.terminal = sys.stdout
        
        # 학습을 resume하는 경우를 위한 예외처리
        if not os.path.exists(filename):
            # terminal.txt 신규생성
            self.log = open(filename, "w")
        else:
            # 기존 terminal.txt 활용
            self.log = open(filename, "a")

        self.filename = filename

    # 메시지를 터미널과 텍스트 파일에 동시에 기록
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    # 출력 스트림을 강제로 비우기 위해 사용
    def flush(self):
        self.terminal.flush()
        self.log.flush()

    # 로그 파일을 닫기 위한 메서드
    def close(self):
        if self.log:
            self.log.close()
            self.log = None

    # 로그 파일을 다시 열기 위한 메서드
    def open(self):
        if not self.log:
            self.log = open(self.filename, "a")

    # 로그 파일을 닫고 다시 여는 메서드
    def save_log(self):
        self.close()
        self.open()

# FileOutputTerminal 클래스를 사용하기 위해서, 아래와 같이 main 함수에 세팅해야 함
if __name__ == "__main__":
    # FileOutputTerminal 클래스를 위한 세팅
    log_filename = "output_log.txt"
    sys.stdout = FileOutputTerminal(log_filename)

    try:
        print("This message will be printed to the console and saved to the log file.")
        print("Another line to be logged.")

        # 로그 파일을 중간에 저장하고 닫았다가 다시 열기
        sys.stdout.save_log()
        print("The log file has been saved, closed, and reopened. This will be appended to the log file.")
    finally:
        # FileOutputTerminal 클래스를 위한 세팅
        # Remember to close the log file properly at the end of your program
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal  # 원래 stdout으로 복원

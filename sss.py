from threading import Thread
import time
from time import sleep

# 自定义的函数，可以替换成其他任何函数
def task(threadName, number, letter):
    print(f"【线程开始】{threadName}\n")
    m = 0
    while m < number:
        sleep(1)
        m += 1
        current_time = time.strftime('%H:%M:%S', time.localtime())
        print(f"[{current_time}] {threadName} 输出 {letter}\n")
    print(f"【线程结束】{threadName}\n")

thread1 = Thread(target=task, args=("thread_1", 6, "a"))  # 线程1：假设任务为打印6个a
thread2 = Thread(target=task, args=["thread_2", 4, "b"])  # 线程2：假设任务为打印4个b
thread3 = Thread(target=task, args=("thread_3", 2, "c"))  # 线程3：假设任务为打印2个c

thread1.start()  # 线程1启动
thread2.start()  # 任务2启动
thread2.join()   # 等待线程2
# thread2.join使得主进程一直在等待thread2线程完成任务，因此直到线程thread2结束后，thread3才开始任务。

thread3.start()  # 线程2完成任务后线程3才启动
thread1.join()   # 等待线程1完成线程
thread3.join()   # 等待线程3完成线程
import numpy as np
import time

def fibonacci(n):
    if n <= 1:
        return n
    return(fibonacci(n - 1) + fibonacci(n - 2))

def main():
    num = np.random.randint(1, 25)
    print(f'{num}th fibonacci number is: {fibonacci(num)}')

start = time.time()
timeout = time.time() + 60 * 2 # Plus 2 minutes
while time.time() <= timeout:
    try:
        main()
        time.sleep(5 - ((time.time() - start) % 5.0)) # 1 minutes interval    
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Existing.')
        exit()
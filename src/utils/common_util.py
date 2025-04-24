import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # 安全处理参数和结果的打印
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
            result_repr = repr(result) if result is not None else "None"
            
            print(f"方法 '{func.__name__}' 执行耗时: {end_time - start_time:.4f} seconds")
            print(f"方法 '{func.__name__}' 执行参数: {', '.join(args_repr + kwargs_repr)}")
            print(f"执行结果: {result_repr}")
            
            return result
        except Exception as e:
            print(f"方法 '{func.__name__}' 执行出错: {str(e)}")
            raise
    return wrapper

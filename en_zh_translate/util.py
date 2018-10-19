# coding:utf-8
"""
this is a demo for utils
"""
import time

def running_time(func):
    """
    获取程序运行时间
    """
    def wrapper(*args, **kwargs):
        print("input", *args)
        local_time = time.time()
        print("running time:%.3f"%(time.time() - local_time))
        print(func(*args))
        #return func(*args, **kwargs)
    return wrapper

a_l = [1,2,3]
a_d = {"a":1,"b":2,"c":"this is a demo"}

@running_time
def test(t_list):
    return t_list[0] + t_list[1]

@running_time
def red(t_list):
    return t_list[0] - t_list[1]


if __name__ =="__main__":
    t_list=[2,1]
    test(t_list)
    red(t_list)

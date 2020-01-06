import numpy as np

def count():
    i = 0.0
    while (i<1000.0):
        i = i+0.5

def count1():
    i = 1000.0
    while (i>0.0):
        i = i-0.5

def sc(a):
    i = 0
    while i < 200:
        count()
        count1()
        i = i+1

    j = 0.0
    while j < 200.0:
        count()
        count1()
        j = j + 0.1
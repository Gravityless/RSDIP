#coding=UTF-8

a, b =  map(int,input("请输入2个数字：").split())

if not(a > b):
    c = a
    a = b
    b = c

x1 = a
x2 = b

while x2:
    c = x2
    x2 = x1 % x2
    x1 = c

print('%d 和 %d 的最大公约数是：%d' % (b, a, x1))


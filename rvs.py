def rvs(s):
    if s=='':
        return s
    else:
        return rvs(s[1:])+s[0]

x=input("Input a String:")
print(rvs(x))

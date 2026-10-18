from typing import Optional


class Dog:
    pass


d: Optional[Dog] = None


def func():
    global d
    if d is not None:
        d = Dog()
    else:
        return d


a = func()

b = func()


print(id(a))
print(id(b))

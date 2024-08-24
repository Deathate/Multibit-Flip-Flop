import dataclasses


@dataclasses.dataclass
class A:
    a: int = 1
    b: list = dataclasses.field(default_factory=list)
    c:str = 'hello'


print(A())

import torch, pickle
class Dummy:
    def __init__(self,*a,**k): pass
    def __setstate__(self,s): self.__dict__.update(s if isinstance(s,dict) else {"state":s})
class U(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("kge"): return type(name,(Dummy,),{})
        return super().find_class(module,name)
class P: Unpickler=U; __name__="P"; load=staticmethod(lambda f,**k: U(f,**k).load())
def load(path): return torch.load(path, map_location="cpu", pickle_module=P)

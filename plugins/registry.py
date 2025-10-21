SOURCES = {}
SINKS = {}

def register_source(name: str):
    def deco(cls):
        SOURCES[name] = cls
        return cls
    return deco

def register_sink(name: str):
    def deco(cls):
        SINKS[name] = cls
        return cls
    return deco

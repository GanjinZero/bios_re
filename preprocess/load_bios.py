import os


class BIOS:
    def __init__(self, path, data_ver='2021-11-05'):
        self.path = path
        self.data_ver = data_ver
        self.load_vocab()
        self.load_type()

    def load_vocab(self):
        with open(os.path.join(self.path, f'Concepts {self.data_ver}.txt'), 'r') as f:
            lines = f.readlines()[1:]
        self.cui2term = {}
        self.term2cui = {}
        self.term2string = {}
        
        for line in lines:
            c, t, s = line.strip().split('\t')
            if c not in self.cui2term:
                self.cui2term[c] = []
            self.cui2term[c].append(t)
            self.term2cui[t] = c
            self.term2string[t] = s

        print(f'Load BIOS @ {self.data_ver}')
        print(f'Concept count: {len(self.cui2term)}')
        print(f'Term count: {len(lines)}')

        return

    def load_type(self):
        with open(os.path.join(self.path, f'Semtypes {self.data_ver}.txt'), 'r') as f:
            lines = f.readlines()[1:]
        self.cui2type = {}
        self.types = set()

        for line in lines:
            c, type = line.strip().split('\t')
            self.cui2type[c] = type.split('|')
            self.types.update(type.split('|'))

        print(f'Type count: {len(self.types)}')

        return

if __name__ == '__main__':
    bios = BIOS('../bios')

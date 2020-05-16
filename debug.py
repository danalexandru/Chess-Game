class Tree(object):

    def __init__(self, value, parent=None, children=[]):
        self.value = value
        self.parent = parent
        self.children = children

    def find(self):
        if len(self.children) == 0:
            yield self
        else:
            for child in self.children:
                yield from child.find_leafs()
    #

    #    def find(self, nodes):
    #        if not hasattr(self, 'children'):
    #            nodes.append(self)
    #        else:
    #            for child in self.children:
    #                child.find(nodes)

    # def find(self):
    #     print('\n----------------------')
    #     print(self.value)
    #     print(self.children)
    #     print('\n----------------------')
    #
    #     if not hasattr(self, 'children'):
    #         return [self]
    #
    #     nodes = []
    #
    #     for child in self.children:
    #         return nodes.extend(child.find())
    #
    #     return nodes


def generate_tree():
    p = Tree(0)
    p.children = [Tree(1, p), Tree(2, p)]

    [c1, c2] = [p.children[0], p.children[1]]
    c1.children = [Tree(3, c1), Tree(4, c1)]
    c2.children = [Tree(5, c2), Tree(6, c2), Tree(7, c2)]

    return p


def find(self):
    if not hasattr(self, 'children'):
        yield self
    else:
        for child in self.children:
            yield from find(child)


def test_child_parent_relation():
    parent_node = Tree(1)
    parent_node.children.append(Tree(2, parent_node, []))

    child_node = parent_node.children[0]

    print(parent_node.value)
    print(child_node.value)

    print(str(parent_node))
    print(str(child_node.parent))


def main():
    # test_child_parent_relation()
    p = generate_tree()
    print(p)
    leafs = p.find()
    # leafs = []
    # p.find(leafs)

    print('leafs = [ ', end='')
    for leaf in leafs:
        print('%d ' % leaf.value, end='')
    print(']')


if __name__ == '__main__':
    main()

import torch as t
from torch import nn
from torch_geometric.nn import conv


class Model(nn.Module):
    def __init__(self, sizes):
        super(Model, self).__init__()

        self.fg = sizes.fg
        self.fd = sizes.fd
        self.k = sizes.k
        self.g = sizes.g
        self.d = sizes.d
        self.gcn_x1 = conv.GCNConv(self.fg, self.fg)
        self.gcn_y1 = conv.GCNConv(self.fd, self.fd)
        self.gcn_x2 = conv.GCNConv(self.fg, self.fg)
        self.gcn_y2 = conv.GCNConv(self.fd, self.fd)

        self.linear_x_1 = nn.Linear(self.fg, 128)
        self.linear_x_2 = nn.Linear(128, 64)
        self.linear_x_3 = nn.Linear(64, 32)

        self.linear_y_1 = nn.Linear(self.fd, 128)
        self.linear_y_2 = nn.Linear(128, 64)
        self.linear_y_3 = nn.Linear(64, 32)

    def forward(self, input):
        t.manual_seed(1)
        x_g = t.randn(self.g, self.fg)
        x_d = t.randn(self.d, self.fd)

        X1 = t.relu(self.gcn_x1(x_g, input[1]['edge_index'], input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]]))
        X = t.relu(self.gcn_x2(X1, input[1]['edge_index'], input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]]))
        X2 = t.relu(self.gcn_x2(X, input[1]['edge_index'], input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]]))

        Y1 = t.relu(self.gcn_y1(x_d, input[0]['edge_index'], input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]]))
        Y = t.relu(self.gcn_y2(Y1, input[0]['edge_index'], input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]]))
        Y2 = t.relu(self.gcn_y2(Y, input[0]['edge_index'], input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]]))

        x1 = t.relu(self.linear_x_1(X2))
        x2 = t.relu(self.linear_x_2(x1))
        x = t.relu(self.linear_x_3(x2))

        y1 = t.relu(self.linear_y_1(Y2))
        y2 = t.relu(self.linear_y_2(y1))
        y = t.relu(self.linear_y_3(y2))

        return x.mm(y.t())

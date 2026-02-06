class Portfolio:
    def __init__(self, initial_cash=100000):
        self.cash = initial_cash
        self.positions = {}
        self.history = []

    def buy(self, symbol, price, shares):
        cost = price * shares
        if self.cash >= cost:
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + shares

    def sell(self, symbol, price, shares):
        if symbol not in self.positions:
            return
        if self.positions[symbol] < shares:
            return
        self.cash += price * shares
        self.positions[symbol] -= shares
        if self.positions[symbol] == 0:
            del self.positions[symbol]

    def value(self, prices):
        total = self.cash
        for s, sh in self.positions.items():
            total += prices.get(s, 0) * sh
        return total

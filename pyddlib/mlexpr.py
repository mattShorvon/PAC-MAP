class MLExpr():

    def __init__(self, expr):
        self.precision = 1e-12
        self._expr = self.reduce(expr)

    @property
    def expr(self):
        return self._expr

    def __repr__(self):
        r = ""
        for term in self.expr:
            sterm = tuple(sorted(term))
            coef = round(self.expr[term], 3)
            if sterm == ():
                r += str(coef) + " "
            else:
                if not (abs(coef - 1) <= self.precision):
                    r += str(coef) + " "
                for i in sterm:
                    r += str(i) + " "
                    # if i > 0:
                    #     r += "x" + str(i) + " "
                    # else:
                    #     r += "y" + str(-i) + " "
                    #     #r += "x̄" + str(-i) + " "
            r += "+ "
        return r[:-3]

    def precision_str(self):
        r = ""
        for term in self.expr:
            sterm = tuple(sorted(term))
            coef = self.expr[term]
            if sterm == ():
                r += str(coef) + " "
            else:
                if not (abs(coef - 1) <= self.precision):
                    r += str(coef) + " "
                for i in sterm:
                    r += str(i) + " "
                    # if i > 0:
                    #     r += "x" + str(i) + " "
                    # else:
                    #     r += "y" + str(-i) + " "
                    #     #r += "x̄" + str(-i) + " "
            r += "+ "
        return r[:-3]

    def reduce(self, expr):
        reduced_expr = {}
        for term in expr:
            sterm = tuple(sorted(term))
            try:
                reduced_expr[sterm] += expr[term]
            except KeyError:
                reduced_expr[sterm] = expr[term]
        dlist = []
        for term in reduced_expr:
            if (abs(reduced_expr[term]) <= self.precision):
                dlist.append(term)
        for term in dlist:
            reduced_expr.pop(term)
        if (len(reduced_expr) == 0):
            reduced_expr[()] = 0.0
        return reduced_expr

    def __bool__(self):
        try:
            return (abs(self.expr[()]) > self.precision) or (len(self.expr) > 1)
        except KeyError:
            return True
            
    def __eq__(self, other):
        if not (isinstance(other, MLExpr)):
            return False
        if (len(self.expr) != len(other.expr)):
            return False
        for term in self.expr:
            if (term not in other.expr):
                return False
            if (abs(self.expr[term] - other.expr[term]) > self.precision):
                return False
        return True

    def __neq__(self, other):
        return not self == other

    def __lt__(self, other):
        x = []
        for i in list(self.expr):
            x.append((i, self.expr[i]))

        y = []
        for i in list(other.expr):
            y.append((i, other.expr[i]))

        x = sorted(x)
        y = sorted(y)
        return x < y

    def __le__(self, other):
        x = []
        for i in list(self.expr):
            x.append((i, self.expr[i]))

        y = []
        for i in list(other.expr):
            y.append((i, other.expr[i]))

        x = sorted(x)
        y = sorted(y)
        return x <= y

    def __gt__(self, other):
        x = []
        for i in list(self.expr):
            x.append((i, self.expr[i]))

        y = []
        for i in list(other.expr):
            y.append((i, other.expr[i]))

        x = sorted(x)
        y = sorted(y)
        return x > y

    def __ge__(self, other):
        x = []
        for i in list(self.expr):
            x.append((i, self.expr[i]))

        y = []
        for i in list(other.expr):
            y.append((i, other.expr[i]))

        x = sorted(x)
        y = sorted(y)
        return x >= y

    # 04:25am, 18/07/2019
    # This was working ok, but I believe that the functions above
    # are more correct

    # def __lt__(self, other):
    #     try:
    #         x = self.expr[()]
    #         y = other.expr[()]
    #     except:
    #         x = sorted(list(self.expr))
    #         y = sorted(list(other.expr))
    #     return x < y

    # def __le__(self, other):
    #     try:
    #         x = self.expr[()]
    #         y = other.expr[()]
    #     except:
    #         x = sorted(list(self.expr))
    #         y = sorted(list(other.expr))
    #     return x <= y

    # def __gt__(self, other):
    #     try:
    #         x = self.expr[()]
    #         y = other.expr[()]
    #     except:
    #         x = sorted(list(self.expr))
    #         y = sorted(list(other.expr))
    #     return x > y

    # def __ge__(self, other):
    #     try:
    #         x = self.expr[()]
    #         y = other.expr[()]
    #     except:
    #         x = sorted(list(self.expr))
    #         y = sorted(list(other.expr))
    #     return x >= y

    def __neg__(self):
        r = {}
        for term in self.expr:
            r[term] = -self.expr[term]
        return MLExpr(r)

    def __add__(self, other):
        r = {}
        for term in self.expr:
            r[term] = self.expr[term]
        for term in other.expr:
            try:
                r[term] += other.expr[term]
            except KeyError:
                r[term] = other.expr[term]
        return MLExpr(r)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        r = {}
        for i in self.expr:
            for j in other.expr:
                r[i+j] = self.expr[i] * other.expr[j]
        return MLExpr(r)

from Infrastructure.utils import Matrix


class SchemeCoefficient(object):
    def __init__(self, inner_row_coefficient: Matrix, between_rows_coefficient: Matrix) -> None:
        self._inner_row_coefficient: Matrix = inner_row_coefficient
        self._between_rows_coefficient: Matrix = between_rows_coefficient

    def dot(self, mat: Matrix) -> Matrix:
        return self._between_rows_coefficient.dot(self._inner_row_coefficient.dot(mat))


class FreeCoefficient(SchemeCoefficient):
    def __init__(self, between_rows_coefficient: Matrix) -> None:
        super(FreeCoefficient, self).__init__(inner_row_coefficient=None,
                                              between_rows_coefficient=between_rows_coefficient)

    def dot(self, mat: Matrix) -> Matrix:
        return self._between_rows_coefficient.dot(mat)

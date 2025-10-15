import numpy as np

class OrthogonalArray:
    def __init__(self, n_factors):
        """
        Construct an orthogonal array for given number of factors using Taguchi L-arrays.
        If n_factors > max template, extend using XOR-based method.
        
        OA = OrthogonalArray(n_factors=20)
        print("Rows:", OA.rows)
        print("OA shape:", OA.data.shape)
        print("OA matrix:\n", OA.data)
        """

        self.rows, base_OA = self._select_base_OA(n_factors)
        base_cols = base_OA.shape[1]
        self.factors = n_factors

        if n_factors <= base_cols:
            self.data = base_OA[:, :n_factors]
        else:
            # Extend using XOR combinations of existing columns
            n_extra = n_factors - base_cols
            extra_cols = []
            for _ in range(n_extra):
                i, j = np.random.choice(base_cols, 2, replace=False)
                new_col = np.bitwise_xor(base_OA[:, i], base_OA[:, j])
                extra_cols.append(new_col)
            extra_cols = np.stack(extra_cols, axis=1)
            self.data = np.concatenate((base_OA, extra_cols), axis=1)

    def RequiredRows(self, n_factors):
        """
        Return the number of rows based on Taguchi L-array size.
        """
        if n_factors <= 3:
            return 4
        elif n_factors <= 7:
            return 8
        elif n_factors <= 11:
            return 12
        elif n_factors <= 15:
            return 16
        else:
            return 32

    def _select_base_OA(self, n_factors):
        """
        Select suitable base Taguchi OA matrix.
        """

        # L4(2^3)
        if n_factors <= 3:
            return 4, np.array([
                [0, 0, 0],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0]
            ])

        # L8(2^7)
        elif n_factors <= 7:
            return 8, np.array([
                [0,0,0,0,0,0,0],
                [0,0,0,1,1,1,1],
                [0,1,1,0,0,1,1],
                [0,1,1,1,1,0,0],
                [1,0,1,0,1,0,1],
                [1,0,1,1,0,1,0],
                [1,1,0,0,1,1,0],
                [1,1,0,1,0,0,1]
            ])

        # L12(2^11)
        elif n_factors <= 11:
            return 12, np.array([
                [0,0,0,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,1,0,0,1],
                [0,1,0,0,1,1,1,1,1,1,0],
                [0,1,1,1,0,0,0,0,1,1,1],
                [1,0,0,1,0,1,0,1,1,0,1],
                [1,0,1,0,1,0,1,0,0,1,1],
                [1,1,0,1,1,0,1,0,1,0,0],
                [1,1,1,0,0,1,0,1,0,1,0],
                [0,0,0,1,1,0,0,1,1,1,0],
                [0,0,1,0,0,1,1,0,0,0,1],
                [0,1,0,1,0,1,1,0,1,0,1],
                [0,1,1,0,1,0,0,1,0,1,1]
            ])

        # L16(2^15)
        elif n_factors <= 15:
            return 16, np.array([
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,1,1,0,0,0,1,1,1,1,1,1,0,0,0],
                [0,1,1,1,1,1,0,0,0,0,0,0,1,1,1],
                [1,0,1,0,1,1,0,0,1,1,1,1,0,0,1],
                [1,0,1,1,0,0,1,1,0,0,0,0,1,1,0],
                [1,1,0,0,1,1,1,1,0,0,1,1,1,1,0],
                [1,1,0,1,0,0,0,0,1,1,0,0,0,0,1],
                [0,0,1,0,0,1,0,1,0,1,0,1,0,1,0],
                [0,0,1,1,1,0,1,0,1,0,1,0,1,0,1],
                [0,1,0,0,1,0,1,0,1,0,1,0,0,1,0],
                [0,1,0,1,0,1,0,1,0,1,0,1,1,0,1],
                [1,0,0,0,1,0,0,1,1,0,0,1,1,0,0],
                [1,0,0,1,0,1,1,0,0,1,1,0,0,1,1],
                [1,1,1,0,0,1,1,0,0,1,1,0,1,0,0],
                [1,1,1,1,1,0,0,1,1,0,0,1,0,1,1]
            ])

        # L32(2^31) – up to 31 factors
        else:
            # Start from L16 and dynamically extend
            base = np.array([
                [0,0,0,0,0],
                [0,0,1,1,1],
                [0,1,0,0,1],
                [0,1,1,1,0],
                [1,0,0,1,0],
                [1,0,1,0,0],
                [1,1,0,1,1],
                [1,1,1,0,1],
                [0,0,0,1,0],
                [0,0,1,0,0],
                [0,1,0,1,1],
                [0,1,1,0,1],
                [1,0,0,0,1],
                [1,0,1,1,1],
                [1,1,0,0,0],
                [1,1,1,1,0]
            ])
            return 16, base
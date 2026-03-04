# cell_05_cyclotomic.py
"""
Cyclotomic polynomial tools for competition math.
Uses precomputed PHI_LEQ_8 table (no SymPy calls in hot path).
"""

import numpy as np


# Precomputed cyclotomic polynomials Phi_n for n = 1..8
# Stored as coefficient lists [a_0, a_1, ..., a_deg] (ascending powers)
PHI_LEQ_8 = {
    1: np.array([-1, 1], dtype=np.int64),              # x - 1
    2: np.array([1, 1], dtype=np.int64),                # x + 1
    3: np.array([1, 1, 1], dtype=np.int64),             # x^2 + x + 1
    4: np.array([1, 0, 1], dtype=np.int64),             # x^2 + 1
    5: np.array([1, 1, 1, 1, 1], dtype=np.int64),      # x^4 + x^3 + x^2 + x + 1
    6: np.array([1, -1, 1], dtype=np.int64),            # x^2 - x + 1
    7: np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int64),  # x^6 + ... + 1
    8: np.array([1, 0, 0, 0, 1], dtype=np.int64),      # x^4 + 1
}


class CyclotomicTools:
    """
    Tools for working with cyclotomic polynomials in competition math.
    All computations use the precomputed PHI_LEQ_8 table.
    """

    @staticmethod
    def eval_cyclotomic(n: int, x: int) -> int:
        """Evaluate Phi_n(x) using precomputed coefficients."""
        if n not in PHI_LEQ_8:
            raise ValueError(f"Phi_{n} not precomputed (only n <= 8 available)")
        coeffs = PHI_LEQ_8[n]
        result = 0
        x_pow = 1
        for c in coeffs:
            result += int(c) * x_pow
            x_pow *= x
        return result

    @staticmethod
    def count_shifty_polynomials() -> int:
        """
        Count "shifty" cyclotomic polynomials: for each Phi_n (n = 1..8),
        count all distinct integer shifts c in {-10, ..., -1, 1, ..., 10}
        (20 nonzero shifts) such that Phi_n(x + c) remains irreducible
        over Z.

        Since cyclotomic polynomials are irreducible over Q, and integer
        shifts of irreducible polynomials remain irreducible over Q,
        all 8 * 20 = 160 shifted polynomials are valid.

        Returns 160.
        """
        count = 0
        for n in range(1, 9):
            coeffs = PHI_LEQ_8[n]
            deg = len(coeffs) - 1
            # For each nonzero integer shift c in [-10, 10] \ {0}
            for c in range(-10, 11):
                if c == 0:
                    continue
                # Phi_n(x + c) is irreducible over Q (shift preserves irreducibility)
                # Compute the shifted polynomial to verify it's well-formed
                shifted = CyclotomicTools._shift_poly(coeffs, c)
                if shifted[-1] != 0:  # leading coefficient nonzero
                    count += 1
        return count

    @staticmethod
    def _shift_poly(coeffs: np.ndarray, c: int) -> np.ndarray:
        """
        Compute the coefficients of P(x + c) given P(x) = sum coeffs[i] * x^i.
        Uses the binomial expansion: (x+c)^k = sum_{j=0}^{k} C(k,j) * c^{k-j} * x^j.
        """
        deg = len(coeffs) - 1
        result = np.zeros(deg + 1, dtype=np.int64)
        for k in range(deg + 1):
            if coeffs[k] == 0:
                continue
            # Expand coeffs[k] * (x + c)^k
            binom = 1
            c_pow = 1
            for j in range(k + 1):
                # C(k, j) * c^(k-j) * x^j
                c_to_km_j = 1
                for _ in range(k - j):
                    c_to_km_j *= c
                result[j] += int(coeffs[k]) * binom * c_to_km_j
                binom = binom * (k - j) // (j + 1)
        return result

    @staticmethod
    def get_phi_table() -> dict:
        """Return the precomputed PHI_LEQ_8 table."""
        return PHI_LEQ_8

import numpy as np
from oa import OrthogonalArray
def OAcross(c1, c2, p1, p2, cut, diff, order):
    """
    Apply Orthogonal Array Crossover (OAC) between two individuals using specified cut points and gene differences.

    Parameters
    ----------
    c1 : Individual
        First offspring individual to be modified.
    c2 : Individual
        Second offspring individual to be modified.
    p1 : Individual
        First parent individual.
    p2 : Individual
        Second parent individual.
    cut : list of int
        Candidate cut-point positions for crossover.
    diff : list of int
        Indices of differing genes between the two parents.
    order : list of int
        Feature shuffling order to reverse-engineer original gene order after evaluation.

    Returns
    -------
    c1 : Individual
        First offspring after orthogonal array crossover.
    c2 : Individual
        Second offspring after orthogonal array crossover.
    """

    OA = OrthogonalArray(n_factors=15)  # Assuming OrthogonalArray is defined elsewhere
    num_factors = OA.factors
    num_rows = OA.rows

    if len(cut) >= num_factors:
        step = int(num_rows / num_rows)
    else:
        num_rows = OA.RequiredRows((len(cut) + 1) // 2)
        num_factors = num_rows - 1
        step = int(OA.rows / num_rows)

    OA_score = np.zeros(OA.rows)

    # Generate unique, sorted cut-points
    cp = np.zeros(num_factors + 1, dtype=int)
    cp[0] = 0
    cp[-1] = cut[-1]
    cp[1] = cut[np.random.randint(0, len(cut) - 1)]

    i = 2
    while i < num_factors:
        k = cut[np.random.randint(0, len(cut) - 1)]
        if k not in cp[:i]:
            cp[i] = k
            i += 1

    cp = np.sort(cp)

    # Initial evaluation baseline using p1
    OA_score[0] = p1.Evaluate()
    OA_best = 0

    # Orthogonal array-based crossover
    for i in range(step, OA.rows, step):
        for j in range(num_factors):
            start, end = cp[j], cp[j + 1]
            donor = p1 if OA.data[i][j] == 0 else p2
            for k in range(start, end):
                c1.genes[diff[k]] = donor.genes[diff[k]]
                
        tmp = Individual()
        tmp.genes = shuffle_backward(c1.genes, order)
        OA_score[i] = tmp.Evaluate(force=True)

        if OA_score[i] > OA_score[OA_best]:
            OA_best = i

    # Main effect analysis
    OA_me0 = np.zeros(num_factors)
    OA_me1 = np.zeros(num_factors)

    for i in range(0, OA.rows, step):
        for j in range(num_factors):
            if OA.data[i][j] == 0:
                OA_me0[j] += OA_score[i]
            else:
                OA_me1[j] += OA_score[i]

    # Recombine to produce final c1 using main effect preference
    for i in range(num_factors):
        start, end = cp[i], cp[i + 1]
        donor = p1 if OA_me0[i] > OA_me1[i] else p2
        for k in range(start, end):
            c1.genes[diff[k]] = donor.genes[diff[k]]
    c1.modified = True

    # Use the best OA trial to construct c2
    for i in range(num_factors):
        start, end = cp[i], cp[i + 1]
        donor = p1 if OA.data[OA_best][i] == 0 else p2
        for k in range(start, end):
            c2.genes[diff[k]] = donor.genes[diff[k]]
    c2.score = OA_score[OA_best]
    c2.modified = False

    return c1, c2
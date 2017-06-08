from sklearn import svm
from sklearn.model_selection import GridSearchCV
import cvxpy as cvx
import numpy as np


def find_hyp_ridge(X, y, C=None, hinge_option='hinge'):
    """
    Determine a separating hyperplane using L2-regularization
    """
    # If no regularization parameter is given, perform a grid search.
    if C is None:
        Cs = [0.001, 0.01, 0.1, 1, 10]
        param_grid = {'C': Cs}
        n_folds = 10
        grid_search = GridSearchCV(
            svm.LinearSVC(loss=hinge_option), param_grid, cv=n_folds)
        grid_search.fit(X, y)
        clf = grid_search.best_estimator_
        C = grid_search.best_params_['C']

    # If a regularization parameter is given, use it.
    else:
        clf = svm.LinearSVC(C=C, loss=hinge_option)
        clf.fit(X, y)

    # Prepare and return the results.
    hyp = np.ndarray.flatten(clf.coef_.T)
    offset = -clf.intercept_
    slack = np.maximum(0, 1 - y * (np.dot(X, hyp) - offset))
    acc = clf.score(X, y)
    return hyp, offset, slack, acc, C


def find_min_relevance(X, y, i, hyp, offset, slack, C, options=None):
    (n, d) = X.shape
    # Calculate constraint values from the original seperating hyperplane.
    L1 = np.linalg.norm(hyp, 1)
    svmloss = np.sum(np.abs(slack))
    # Define optimization problem variables.
    xp = cvx.Variable()  # x' , our opt. value
    omega = cvx.Variable(d)  # complete linear weight vector
    b = cvx.Variable()  # shift
    eps = cvx.Variable(n)

    # Define minimizing optimization problem.
    objective_min = cvx.Minimize(xp)
    constraints_min = [
        # Use slack variables.
        cvx.mul_elemwise(y.T, X * omega - b) >= 1 - eps,
        eps >= 0,
        # Control L1 norm. and slack.
        cvx.norm(omega, 1) <= L1,
        cvx.sum_entries(eps) <= svmloss,
        # Put the correct bound on objective.
        cvx.abs(omega[i]) <= xp
    ]

    problem_min = cvx.Problem(objective_min, constraints_min)
    problem_min.solve()

    # Ensure that the problem was solved properly.
    if problem_min.status != cvx.OPTIMAL:
        raise RuntimeError(
            'Minimization problem for feature {} not properly solved. Status: {}'.
            format(i, problem_min.status))

    return omega.value, b.value, eps.value, xp.value


def find_max_relevance_pos(X, y, i, hyp, offset, slack, C, options=None):
    (n, d) = X.shape
    # Calculate contraint values from the original seperating hyperplane.
    L1 = np.linalg.norm(hyp, 1)
    svmloss = np.sum(np.abs(slack))
    # Define optimization problem variables.
    xp = cvx.Variable()  # x' , our opt. value
    omega = cvx.Variable(d)  # complete linear weight vector
    b = cvx.Variable()  # shift
    eps = cvx.Variable(n)

    objective_max = cvx.Maximize(xp)
    constraints_max_pos = [
        # Use slack variables.
        cvx.mul_elemwise(y.T, X * omega - b) >= 1 - eps,
        eps >= 0,
        # Control L1 norm. and slack.
        cvx.norm(omega, 1) <= L1,
        cvx.sum_entries(eps) <= svmloss,
        # Consider only normal vectors where omega[i] is positive.
        xp <= omega[i]
    ]

    problem_max_pos = cvx.Problem(objective_max, constraints_max_pos)
    problem_max_pos.solve()

    # Ensure that the problem was solved properly.
    if problem_max_pos.status != cvx.OPTIMAL:
        raise RuntimeError(
            'Positive maximization problem for feature {} not properly solved. Status: {}'.
            format(i, problem_max_pos.status))

    return omega.value, b.value, eps.value, xp.value


def find_max_relevance_neg(X, y, i, hyp, offset, slack, C, options=None):
    (n, d) = X.shape
    # Calculate contraint values from the original seperating hyperplane.
    L1 = np.linalg.norm(hyp, 1)
    svmloss = np.sum(np.abs(slack))
    # Define optimization problem variables.
    xp = cvx.Variable()  # x' , our opt. value
    omega = cvx.Variable(d)  # complete linear weight vector
    b = cvx.Variable()  # shift
    eps = cvx.Variable(n)

    objective_max = cvx.Maximize(xp)
    constraints_max_neg = [
        # Use slack variables.
        cvx.mul_elemwise(y.T, X * omega - b) >= 1 - eps,
        eps >= 0,
        # Control L1 norm. and slack.
        cvx.norm(omega, 1) <= L1,
        cvx.sum_entries(eps) <= svmloss,
        # Consider only normal vectors where omega[i] is negative.
        xp <= -omega[i]
    ]

    problem_max_neg = cvx.Problem(objective_max, constraints_max_neg)
    problem_max_neg.solve()

    # Ensure that the problem was solved properly.
    if problem_max_neg.status != cvx.OPTIMAL:
        raise RuntimeError(
            'Negative maximization problem for feature {} not properly solved. Status: {}'.
            format(i, problem_max_neg.status))

    return omega.value, b.value, eps.value, xp.value


def find_max_relevance(X, y, i, hyp, offset, slack, C, options=None):
    # Select the optimization problem with the better result.
    omega_pos, b_pos, eps_pos, xp_pos = find_max_relevance_pos(
        X, y, i, hyp, offset, slack, C, options)
    omega_neg, b_neg, eps_neg, xp_neg = find_max_relevance_neg(
        X, y, i, hyp, offset, slack, C, options)
    if xp_neg > xp_pos:
        return omega_neg, b_neg, eps_neg, xp_neg
    else:
        return omega_pos, b_pos, eps_pos, xp_pos


def find_relevances(X, y, hyp, offset, slack, C, options=None):
    L1 = np.linalg.norm(hyp, 1)
    svmloss = np.sum(np.abs(slack))
    (n, d) = X.shape
    # Initialize arrays for optimization results.
    omegas = np.zeros(
        [d, 2 * d])  # The normal vector, a min and max one for each dim
    bs = np.zeros(2 * d)  # The offsets
    xps = np.zeros(2 * d)  # The extreme results
    print('Original weight vector L1 is {} and slack variable sum is {}.'.
          format(L1, svmloss))
    for i in range(d):
        omega, b, eps, xp = find_min_relevance(X, y, i, hyp, offset, slack, C,
                                               options)
        omegas[:, 2 * i] = np.asarray(omega).reshape((d, ))
        bs[2 * i] = b
        xps[2 * i] = xp
        print(
            'Found min weight {} for feature {} in vector with L1-norm {}. Slack sum is {}.'.
            format(xp, i,
                   np.linalg.norm(np.asarray(omega), 1),
                   np.sum(np.abs(np.asarray(eps)))))
        omega, b, eps, xp = find_max_relevance(X, y, i, hyp, offset, slack, C,
                                               options)
        omegas[:, 2 * i + 1] = np.asarray(omega).reshape((d, ))
        bs[2 * i + 1] = b
        xps[2 * i + 1] = xp
        print(
            'Found max weight {} for feature {} in vector with L1-norm {}. Slack sum is {}.'.
            format(xp, i,
                   np.linalg.norm(np.asarray(omega), 1),
                   np.sum(np.abs(np.asarray(eps)))))
    return omegas, xps, bs

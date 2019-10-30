from sklearn import svm
from sklearn.model_selection import GridSearchCV
import cvxpy as cvx
import numpy as np
import sklearn.utils

def find_hyp_ridge(X, y, C=None, hinge_option='hinge', random_state=None):
    """
    Determine a separating hyperplane using L2-regularization
    """
    # If no regularization parameter is given, perform a grid search.
    if C is None:
        Cs = [0.001, 0.01, 0.1, 1, 10]
        param_grid = {'C': Cs}
        n_folds = 10
        grid_search = GridSearchCV(
            svm.LinearSVC(loss=hinge_option, random_state=random_state), param_grid, cv=n_folds)
        grid_search.fit(X, y)
        clf = grid_search.best_estimator_
        C = grid_search.best_params_['C']

    # If a regularization parameter is given, use it.
    else:
        clf = svm.LinearSVC(C=C, loss=hinge_option, random_state=random_state)
        clf.fit(X, y)

    # Prepare and return the results.
    hyp = np.ndarray.flatten(clf.coef_.T)
    offset = -clf.intercept_
    slack = np.maximum(0, 1 - y * (np.dot(X, hyp) - offset))
    acc = clf.score(X, y)
    return hyp, offset, slack, acc, C


def find_hyp_l1(X, y, C, random_state=None):
    """
    Determine a separating hyperplane using L1-regularization
    """
    # Prepare variables.
    (n, d) = X.shape
    w = cvx.Variable(d)
    xi = cvx.Variable(n)
    b = cvx.Variable()
    
    # Prepare problem.
    objective = cvx.Minimize(cvx.norm(w, 1) + C * cvx.sum(xi))
    constraints = [
        cvx.multiply(y.T, X * w - b) >= 1 - xi,
        xi >= 0
    ]
    # Solve problem.
    problem = cvx.Problem(objective, constraints)
    problem.solve()
    
    # Prepare output and convert from matrices to flattened arrays.
    hyp = np.asarray(w.value).flatten()
    offset = b.value
    slack = np.asarray(xi.value).flatten()
    return hyp, offset, slack, C



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
        cvx.multiply(y.T, X * omega - b) >= 1 - eps,
        eps >= 0,
        # Control L1 norm. and slack.
        cvx.norm(omega, 1) <= L1,
        cvx.sum(eps) <= svmloss,
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
        cvx.multiply(y.T, X * omega - b) >= 1 - eps,
        eps >= 0,
        # Control L1 norm. and slack.
        cvx.norm(omega, 1) <= L1,
        cvx.sum(eps) <= svmloss,
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
        cvx.multiply(y.T, X * omega - b) >= 1 - eps,
        eps >= 0,
        # Control L1 norm. and slack.
        cvx.norm(omega, 1) <= L1,
        cvx.sum(eps) <= svmloss,
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


def find_shadow_relevances(X, y, hyp, offset, slack, C, random_state=None, options=None):
    L1 = np.linalg.norm(hyp, 1)
    svmloss = np.sum(np.abs(slack))
    (n, d) = X.shape
    random_state = sklearn.utils.check_random_state(random_state)
    # Initialize arrays for optimization results.
    xps = np.zeros(d)  # The extreme results
    for i in range(d):
        X_shadow = np.append(
            X, X[random_state.permutation(n), i].reshape((n, 1)), axis=1)
        xp = find_max_relevance(X_shadow, y, d, hyp, offset, slack, C,
                                options)[3]
        xps[i] = xp
    return xps

def find_relevances(X, y, hyp, offset, slack, C, options=None):
    L1 = np.linalg.norm(hyp, 1)
    svmloss = np.sum(np.abs(slack))
    (n, d) = X.shape
    # Initialize arrays for optimization results.
    omegas = np.zeros(
        [d, 2 * d])  # The normal vector, a min and max one for each dim
    bs = np.zeros(2 * d)  # The offsets
    xps = np.zeros(2 * d)  # The extreme results
    for i in range(d):
        omega, b, eps, xp = find_min_relevance(X, y, i, hyp, offset, slack, C,
                                               options)
        omegas[:, 2 * i] = np.asarray(omega).reshape((d, ))
        bs[2 * i] = b
        xps[2 * i] = xp
        omega, b, eps, xp = find_max_relevance(X, y, i, hyp, offset, slack, C,
                                               options)
        omegas[:, 2 * i + 1] = np.asarray(omega).reshape((d, ))
        bs[2 * i + 1] = b
        xps[2 * i + 1] = xp
    return omegas, xps, bs

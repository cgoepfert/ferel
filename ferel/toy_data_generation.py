import numpy as np
import sklearn.utils
from sklearn.datasets import make_regression


def gen_strongly_relevant(n_features,
                          normal_vector,
                          n_samples=100,
                          b=0,
                          margin=1,
                          data_range=10,
                          random_state=None):
    """ Generate data uniformly distributed in a square and perfectly separated by the hyperplane given by normal_vector and b.

    Keyword arguments:
    n_samples -- number of samples required (default 100)
    n_features -- number of features required
    normal_vector -- the normal vector of the separating hyperplane
    b -- the offset of the separating hyperplane defined by normal_vector * x - b = 0 (default 0)
    margin -- intrusion-free margin of the optimal separating hyperplane (default 0.2)
    data_range -- data is distributed between -data_range and data_range (default 10)
    """

    # Check random state.
    random_state = sklearn.utils.check_random_state(random_state)

    # Sample data uniformly.
    data = random_state.uniform(-data_range, data_range,
                                (n_samples, int(n_features))) + b

    # Re-roll margin intrusions.
    intruders = np.abs(np.inner(normal_vector, data) - b) < margin
    while np.sum(intruders) > 0:
        data[intruders] = random_state.uniform(
            -data_range, data_range, (np.sum(intruders), int(n_features))) + b
        intruders = np.abs(np.inner(normal_vector, data) - b) < margin

    # Label data according to placement relative to the hyperplane induced by normal_vector and b.
    labels = np.ones(n_samples)
    labels[np.inner(normal_vector, data) - b > 0] = 1
    labels[np.inner(normal_vector, data) - b < 0] = -1
    return data, labels


def gen_weakly_relevant(strongly_relevant_data,
                        normal_vector,
                        random_state=None):
    """ For each column of strongly_relevant_data and normal_vector, create a copy and randomly scale the data with coefficients between zero and one, while the normal vector is inversely scaled. 

    Keyword arguments:
    strongly_relevant_data -- the data
    normal_vector -- the normal vector
    """

    # Check random state.
    random_state = sklearn.utils.check_random_state(random_state)
    n_features = np.shape(strongly_relevant_data)[1]
    assert len(normal_vector) == n_features

    # Duplicate each feature.
    mask = np.arange(2 * n_features)
    mask = np.floor(mask / 2).astype(int)
    weakly_relevant_data = strongly_relevant_data[:, mask]
    normal_vector = normal_vector[mask]

    # Rescale features by random coefficients.
    coef = random_state.uniform(size=2 * n_features)
    weakly_relevant_data = np.dot(weakly_relevant_data, np.diag(coef))
    normal_vector = normal_vector / coef
    return weakly_relevant_data, normal_vector


def gen_irrelevant(n_features, n_samples=100, data_range=10,
                   random_state=None):
    """Return uniformly distributed random data with absolute value at most data_range."""

    # Check random state.
    random_state = sklearn.utils.check_random_state(random_state)
    return random_state.uniform(-data_range, data_range, (n_samples,
                                                          int(n_features)))


def gen_toy_data_uniform(n_samples=100,
                         n_str_rel=1,
                         n_weak_rel=0,
                         n_irrel=1,
                         margin=1,
                         random_state=None,
                         min_relevance=0.2,
                         data_range=10):
    """ Generate uniformly distributed toy data that can be perfectly separated by a hyperplane and return data, labels in {-1, 1} and the normal vector of the separating hyperplane.

    Keyword arguments:
    n_samples -- number of samples (default 100)
    n_str_rel -- number of strongly relevant features (default 1)
    n_weak_rel -- number of weakly relevant features, must be a multiple of two (default 2)
    n_irrel -- number of irrelevant features (default 1)
    margin -- intrusion-free margin of the optimal separating hyperplane (default 0.2)
    random_state -- random state used for generating data
    min_relevance -- minimal entry of the relevance vector (default 0.2)
    data_range -- data is distributed between -data_range and data_range (default 10)
    """

    if not 0 < n_samples:
        raise ValueError("We need at least one sample.")
    if not 0 < n_str_rel + n_weak_rel + n_irrel:
        raise ValueError("We need at least one feature.")
    if not n_weak_rel % 2 == 0:
        raise ValueError("Number of redundant features has to be even.")
    if not min_relevance < 1:
        raise ValueError("Minimal relevance has to be smaller than one.")

    # Check random state.
    random_state = sklearn.utils.check_random_state(random_state)
    n_features = n_str_rel + n_weak_rel + n_irrel
    data = np.zeros((n_samples, n_features))

    # Generate random relevances for strongly and weakly relevant features.
    normal_vector = random_state.uniform(
        min_relevance, 1, int(n_str_rel + n_weak_rel / 2)
    ) * random_state.choice([1, -1], int(n_str_rel + n_weak_rel / 2))
    b = random_state.uniform(-1, 1)

    # If strongly or weakly relevant features exist, generate an appropriate number of them.
    if n_str_rel + n_weak_rel / 2 > 0:
        strongly_relevant_data, labels = gen_strongly_relevant(
            n_str_rel + n_weak_rel / 2,
            normal_vector,
            n_samples,
            b,
            margin,
            data_range,
            random_state=random_state)

    # Write features into data and relevances into normal vector.
    if n_str_rel > 0:
        data[:, :n_str_rel] = strongly_relevant_data[:, :n_str_rel]
    if n_weak_rel > 0:
        data[:, n_str_rel:
             n_str_rel + n_weak_rel], weak_normal_vector = gen_weakly_relevant(
                 strongly_relevant_data[:, n_str_rel:],
                 normal_vector[n_str_rel:],
                 random_state=random_state)
        normal_vector = np.concatenate((normal_vector[:n_str_rel],
                                        weak_normal_vector))
    if n_irrel > 0:
        data[:, n_str_rel + n_weak_rel:] = gen_irrelevant(
            n_irrel, n_samples, data_range, random_state=random_state)
        normal_vector = np.concatenate((normal_vector, np.zeros(n_irrel)))
    return data, labels, normal_vector, b


def add_label_noise_uniform(y, p, random_state=None):
    """ Flip percentage p of labels in y and return disturbed labels. """

    # Check random state.
    random_state = sklearn.utils.check_random_state(random_state)

    swap = random_state.rand(len(y)) < p
    noisy = np.copy(y)
    noisy[swap] = -1 * y[swap]
    return noisy


def add_label_noise_border(X, y, normal_vector, b, p, random_state=None):
    """ Flip labels in y with growing probability the closer the corresponding point is to the hyperplane defined by normal vector and b and return disturbed labels. """

    # Check random state.
    random_state = sklearn.utils.check_random_state(random_state)

    swap = random_state.rand(len(y)) * y * (np.dot(X, normal_vector) - b) < p
    noisy = np.copy(y)
    noisy[swap] = -1 * y[swap]
    return noisy


def add_data_noise_gaussian(X, stdev, random_state=None):
    """ Add random gaussian noise with mean 0 and std stdev to data X and return X. """

    # Check random state.
    random_state = sklearn.utils.check_random_state(random_state)

    X = X + stdev * random_state.randn(X.shape())
    return X

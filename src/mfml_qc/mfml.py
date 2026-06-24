import numpy as np
import time
import copy
from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor


from .krr import (
    KRR,
    gaussian_kernel_symmetric,
    gaussian_kernel_asymmetric,
    matern_kernel_symmetric,
    matern_kernel_asymmetric,
    laplacian_kernel_symmetric,
    laplacian_kernel_asymmetric,
    wasserstein_kernel_symmetric,
    wasserstein_kernel_asymmetric,
)
from .utils import property_differences


class ModelMFML:
    """
    The Multi-Fidelity Machine Learning (MFML) model.

    This class carries out the training and prediction of
    MFML models. It supports both standard MFML and the
    optimzied MFML (o-MFML) models which are data-adaptive
    combinations of the sub-models.
    """

    def __init__(
        self,
        reg: float = 1e-9,
        kernel: str = "matern",
        sigma: float = 715.0,
        nu: float = 1.5,
        p: float = 1.0,
        q: float = 1.0,
        p_bar: bool = False,
        base_estimator: object = None,
    ):
        """
        Initializes the MFML model class.

        Parameters
        ----------
        reg : float, optional
            Regularization parameter for the built-in KRR. Defaults to 1e-9.
        kernel : str, optional
            Kernel type ('matern', 'gaussian', 'laplacian', 'wasserstein', 'linear').
            Defaults to "matern".
        sigma : float, optional
            Kernel width parameter for the default KRR estimator. Defaults to 715.0.
        nu : float, optional
            Smoothness parameter for the Matérn kernel (0.5, 1.5, 2.5). Defaults to 1.5.
        p : float, optional
            Power parameter for the Wasserstein kernel. Defaults to 1.0.
        q : float, optional
            Outer exponent parameter for the Wasserstein kernel. Defaults to 1.0.
        p_bar : bool, optional
            Enables or disables the tqdm progress bars during training and prediction.
            Defaults to False.
        base_estimator : object, optional
            A custom ML model instance to use (e.g., from scikit-learn).
            If None, defaults to the built-in KRR. Must have a `.fit(X, y)`
            or `.train(X, y)` method, and a `.predict(X)` method.
        """
        self.reg = reg
        self.kernel = kernel
        self.sigma = sigma
        self.nu = nu
        self.p = p
        self.q = q
        self.base_estimator = base_estimator

        # Data params
        self.X_train_parent = None
        self.X_trains = None
        self.y_trains = None
        self.indexes = None

        # Model storage
        self.models = None
        self.LCCoptimizer = None
        self.coeffs = None

        # Score params
        self.mae = 0.0
        self.rmse = 0.0
        self.train_time = 0.0
        self.predict_time = 0.0
        self.p_bar = p_bar

    def _generate_nested_indexes(self, n_trains=None, shuffle=False, seed=0):
        """
        Subsets the data indexes to match specified training set sizes while
        strictly retaining the nested multifidelity structure.

        Uses a bottom-up approach:
        1. Selects from the lowest fidelity (baseline).
        2. For subsequent higher fidelities, selects ONLY from the subset
           chosen in the previous fidelity.

        Parameters
        ----------
        n_trains : np.ndarray, optional
            Array specifying the target number of training samples for each fidelity.
            If None, uses all available samples.
        shuffle : bool, optional
            If True, randomizes the selection deterministically based on the seed.
            If False, sequentially selects the first valid nested matches.
        seed : int, optional
            Random seed used for shuffling. Defaults to 0.

        Returns
        -------
        np.ndarray
            An object array of shape (nfids,) containing the patched index mappings
            for each fidelity level.
        """
        import warnings

        nfids = self.indexes.shape[0]
        if n_trains is None:
            n_trains = np.asarray([self.indexes[i].shape[0] for i in range(nfids)])

        subset_index_array = np.zeros((nfids), dtype=object)

        # set seed
        rng = np.random.RandomState(seed) if shuffle else None

        # Tracks the selected baseline IDs from the previous (lower) fidelity
        previous_selected_b_ids = None

        for i in range(nfids):
            avail_b_ids = self.indexes[i][:, 0]

            if i == 0:
                # For the baseline, candidates are everything available
                candidates = list(avail_b_ids)
            else:
                # For higher fidelities, candidates MUST exist in the previous (lower) fidelity's selection
                prev_set = set(previous_selected_b_ids)
                candidates = [b for b in avail_b_ids if b in prev_set]

            needed = n_trains[i]

            # fallback if user requests more samples than exist within the strict nesting constraints
            if needed > len(candidates):
                warnings.warn(
                    f"Requested {needed} samples for fidelity {i}, but only {len(candidates)} "
                    f"are available within the nested baseline subset. Truncating to {len(candidates)}.",
                    UserWarning,
                )
                needed = len(candidates)

            if needed > 0:
                if shuffle:
                    rng.shuffle(candidates)
                selected_b_ids = candidates[:needed]
            else:
                selected_b_ids = []

            previous_selected_b_ids = selected_b_ids

            # Map the selected baseline IDs back to [baseline_id, level_id] for this fidelity
            fid_map = {row[0]: row[1] for row in self.indexes[i]}

            patched_ind = []
            for b_idx in selected_b_ids:
                patched_ind.append([b_idx, fid_map[b_idx]])

            # Sort to ensure consistent row ordering across all fidelities
            patched_ind.sort(key=lambda x: x[0])
            subset_index_array[i] = np.asarray(patched_ind, dtype=int)

        return subset_index_array

    def y_train_breakup(self):
        """
        Extracts the target property arrays (y) for the required
        multifidelity sub-models.

        For N fidelities, the MFML method requires 2N - 1 sub-models:
        N models trained on the target properties directly (upper), and
        N - 1 models trained on the lower fidelity representations of the
        higher fidelity subsets (lower).
        """
        n = self.indexes.shape[0]
        y_trains = np.zeros((2 * n - 1), dtype=object)
        count = 0

        for i in tqdm(range(n), desc="Extracting upper y_trains", leave=self.p_bar):
            ind_i = self.indexes[i][:, 1]
            y_trains[count] = np.copy(self.y_trains[i][ind_i])
            count += 1

        for i in tqdm(range(n - 1), desc="Extracting lower y_trains", leave=self.p_bar):
            ind_i = self.indexes[i]
            ind_ip1 = self.indexes[i + 1]
            c_i = []
            for row in ind_ip1:
                temp_i = np.where(ind_i[:, 0] == row[0])[0]
                if np.size(temp_i) != 0:
                    c_i.append(ind_i[temp_i[0], 1])
            y_trains[count] = np.copy(self.y_trains[i][np.asarray(c_i)])
            count += 1

        self.y_trains = y_trains

    def X_train_breakup(self):
        """
        Extracts the feature matrices (X) for each fidelity level.

        Slices the master `X_train_parent` array using the parsed nested indexes
        so that each fidelity level has a corresponding, correctly sized feature matrix.
        """
        n = self.indexes.shape[0]
        X_trains = np.zeros((n), dtype=object)
        for i in tqdm(range(n), desc="Extracting X_trains", leave=self.p_bar):
            ind_i = self.indexes[i][:, 0]
            X_trains[i] = self.X_train_parent[ind_i]
        self.X_trains = np.copy(X_trains)

    def _get_optimizer_kernel(self, X1, X2, ktype, sigma, order_nu, metric_p):
        """
        Helper method to evaluate specific kernel matrices for the KRR/CompKRR optimizers
        in the o-MFML model. This helper function is also used in the non-linear formulation
        of MFML.

        Parameters
        ----------
        X1 : np.ndarray
            First input feature matrix.
        X2 : np.ndarray or None
            Second input feature matrix. If None, computes a symmetric kernel.
        ktype : str
            Kernel type ('matern', 'gaussian', 'laplacian', 'wasserstein', 'linear').
        sigma : float
            Kernel width parameter.
        order_nu : float
            Smoothness parameter for Matérn kernel.
        metric_p : float
            Power parameter for Wasserstein kernel.

        Returns
        -------
        np.ndarray
            The computed kernel matrix.
        """
        if ktype == "gaussian":
            return (
                gaussian_kernel_symmetric(X1, sigma)
                if X2 is None
                else gaussian_kernel_asymmetric(X1, X2, sigma)
            )
        elif ktype == "laplacian":
            return (
                laplacian_kernel_symmetric(X1, sigma)
                if X2 is None
                else laplacian_kernel_asymmetric(X1, X2, sigma)
            )
        elif ktype == "matern":
            return (
                matern_kernel_symmetric(X1, sigma, order_nu)
                if X2 is None
                else matern_kernel_asymmetric(X1, X2, sigma, order_nu)
            )
        elif ktype == "wasserstein":
            return (
                wasserstein_kernel_symmetric(X1, sigma, order_nu, metric_p)
                if X2 is None
                else wasserstein_kernel_asymmetric(X1, X2, sigma, order_nu, metric_p)
            )
        else:
            # Linear kernel fallback
            return np.dot(X1, X1.T) if X2 is None else np.dot(X2, X1.T)

    def _instantiate_and_train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Helper method to securely instantiate and train a sub-model.

        Uses duck typing to support arbitrary model architectures (e.g., standard
        scikit-learn estimators via `.fit` or the custom KRR via `.train`).

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target array.

        Returns
        -------
        object
            The trained model instance.
        """
        if self.base_estimator is None:
            model = KRR(
                kernel_type=self.kernel,
                sigma=self.sigma,
                nu=self.nu,
                p=self.p,
                q=self.q,
                reg=self.reg,
            )
        else:
            model = copy.deepcopy(self.base_estimator)

        # Support both (.train) and (.fit)
        if hasattr(model, "train"):
            model.train(X_train, y_train)
        elif hasattr(model, "fit"):
            model.fit(X_train, y_train)
        else:
            raise AttributeError(
                "The provided base_estimator must have either a '.train(X, y)' or '.fit(X, y)' method."
            )

        return model

    def train(
        self,
        X_train_parent: np.ndarray,
        file_paths: list = None,
        y_trains: np.ndarray = None,
        indexes: np.ndarray = None,
        shuffle: bool = False,
        n_trains: np.ndarray = None,
        seed: int = 0,
    ):
        """
        Multifidelity data extraction and training of the sub-models.

        Parameters
        ----------
        X_train_parent : np.ndarray
            The complete feature matrix corresponding to the baseline (lowest fidelity) data.
        file_paths : list of str, optional
            List of paths to property files ordered from lowest to highest fidelity.
            Required if `y_trains` and `indexes` are not provided.
        y_trains : np.ndarray, optional
            Precomputed object array of target properties for each fidelity.
        indexes : np.ndarray, optional
            Precomputed object array of nested mapping indexes.
        shuffle : bool, optional
            If True, randomly shuffles the selected nested subsets. Defaults to False.
        n_trains : np.ndarray, optional
            Array specifying the target number of training samples for each fidelity.
        seed : int, optional
            Random seed for shuffling. Defaults to 0.

        Raises
        ------
        ValueError
            If neither precomputed arrays (`y_trains`, `indexes`) nor `file_paths` are provided.
        """
        tstart = time.time()
        self.X_train_parent = np.copy(X_train_parent)

        if y_trains is None and indexes is None:
            if file_paths is None:
                raise ValueError(
                    "Must provide either precomputed y_trains/indexes or file_paths."
                )
            self.y_trains, self.indexes = property_differences(file_paths)
        else:
            self.y_trains = y_trains
            self.indexes = indexes

        nfids = self.indexes.shape[0]

        # generate indexes/ shuffle as needed
        self.indexes = self._generate_nested_indexes(
            n_trains=n_trains, shuffle=shuffle, seed=seed
        )

        self.X_train_breakup()
        self.y_train_breakup()

        self.models = np.zeros((2 * nfids - 1), dtype=object)

        # keeps track of the sub-models
        count = 0
        # Upper training
        for i in tqdm(
            range(nfids), desc="Training upper ML models...", leave=self.p_bar
        ):
            self.models[count] = self._instantiate_and_train(
                self.X_trains[i], self.y_trains[count]
            )
            count += 1

        # Lower training
        for i in tqdm(
            range(nfids - 1), desc="Training lower ML models", leave=self.p_bar
        ):
            self.models[count] = self._instantiate_and_train(
                self.X_trains[i + 1], self.y_trains[count]
            )
            count += 1

        self.train_time = time.time() - tstart

    def predict(
        self,
        X_test: np.ndarray,
        X_val: np.ndarray = None,
        y_test: np.ndarray = None,
        y_val: np.ndarray = None,
        optimiser: str = "default",
        **optargs,
    ):
        """
        Predicts target values using the trained multifidelity ensemble.

        Supports standard Single-Grid Combination Technique (SGCT) arithmetic or
        advanced machine-learned combinations (o-MFML) using a validation set.

        Parameters
        ----------
        X_test : np.ndarray
            The testing feature matrix.
        X_val : np.ndarray, optional
            Validation feature matrix, required if using an advanced optimizer.
        y_test : np.ndarray, optional
            True target values for the test set. If provided, computes MAE and RMSE
            and saves them to the model object.
        y_val : np.ndarray, optional
            True target values for the validation set, required if using an advanced optimizer.
        optimiser : str, optional
            The combination strategy to use. Options include: 'default' (SGCT),
            'OLS', 'LRR', 'LASSO', 'MLPR', 'KRR', or 'CompKRR'. Defaults to 'default'.
        **optargs : dict
            Additional hyperparameters to pass to the chosen optimizer model.

        Returns
        -------
        np.ndarray
            The final predicted target values for the test set.
        """
        tstart = time.time()
        nfids = self.indexes.shape[0]

        test_preds = np.zeros((X_test.shape[0], 2 * nfids - 1), dtype=float)

        # instantiate validation predictions if required
        # only if y_val is given since we use y_val in the optimizations
        if y_val is not None:
            val_preds = np.zeros((X_val.shape[0], 2 * nfids - 1), dtype=float)

        count = 0
        # Upper triangle preds
        for i in tqdm(range(nfids), desc="Upper MFML predictions", leave=self.p_bar):
            if y_val is not None:
                val_preds[:, count] = self.models[count].predict(X_val)
            test_preds[:, count] = self.models[count].predict(X_test)
            count += 1

        # Lower triangle preds
        for i in tqdm(
            range(nfids - 1), desc="Lower MFML predictions", leave=self.p_bar
        ):
            if y_val is not None:
                val_preds[:, count] = self.models[count].predict(X_val)
            test_preds[:, count] = self.models[count].predict(X_test)
            count += 1

        # optimzers for o-MFML
        if optimiser == "OLS":
            defaultKwargs = {"copy_X": True, "fit_intercept": False}
            defaultKwargs.update(**optargs)
            regressor = LinearRegression(**defaultKwargs)
            regressor.fit(val_preds, y_val)
            final_preds = regressor.predict(test_preds)
            self.LCCoptimizer = regressor

        elif optimiser == "LRR":
            defaultKwargs = {"alpha": 1e-9, "fit_intercept": False, "copy_X": True}
            defaultKwargs.update(**optargs)
            regressor = Ridge(**defaultKwargs)
            regressor.fit(val_preds, y_val)
            final_preds = regressor.predict(test_preds)
            self.LCCoptimizer = regressor

        elif optimiser == "LASSO":
            defaultKwargs = {"alpha": 1.0, "fit_intercept": False, "max_iter": 1000}
            defaultKwargs.update(**optargs)
            regressor = Lasso(**defaultKwargs)
            regressor.fit(val_preds, y_val)
            final_preds = regressor.predict(test_preds)
            self.LCCoptimizer = regressor

        elif optimiser == "MLPR":
            defaultKwargs = {
                "hidden_layer_sizes": (100,),
                "activation": "relu",
                "solver": "adam",
            }
            defaultKwargs.update(**optargs)
            MLPR = MLPRegressor(**defaultKwargs)
            MLPR.fit(val_preds, y_val)
            final_preds = MLPR.predict(test_preds)
            self.LCCoptimizer = MLPR

        elif optimiser == "KRR":
            defaultKwargs = {
                "sigma": 700.0,
                "reg": 1e-9,
                "kernel_type": "gaussian",
                "order": 1.5,
                "metric": 1.0,
            }
            defaultKwargs.update(**optargs)

            K_val = self._get_optimizer_kernel(
                val_preds,
                None,
                defaultKwargs["kernel_type"],
                defaultKwargs["sigma"],
                defaultKwargs["order"],
                defaultKwargs["metric"],
            )
            K_eval = self._get_optimizer_kernel(
                val_preds,
                test_preds,
                defaultKwargs["kernel_type"],
                defaultKwargs["sigma"],
                defaultKwargs["order"],
                defaultKwargs["metric"],
            )

            K_val[np.diag_indices_from(K_val)] += defaultKwargs["reg"]
            opt_alpha = np.linalg.solve(K_val, y_val)
            final_preds = np.dot(K_eval, opt_alpha)
            self.coeffs = opt_alpha

        elif optimiser == "CompKRR":
            defaultKwargs = {
                "sigma": 700.0,
                "reg": 1e-9,
                "kernel_type": "gaussian",
                "order": 1.5,
                "metric": 1.0,
            }
            defaultKwargs.update(**optargs)

            K_val = self._get_optimizer_kernel(
                val_preds,
                None,
                defaultKwargs["kernel_type"],
                defaultKwargs["sigma"],
                defaultKwargs["order"],
                defaultKwargs["metric"],
            )
            K_eval = self._get_optimizer_kernel(
                val_preds,
                test_preds,
                defaultKwargs["kernel_type"],
                defaultKwargs["sigma"],
                defaultKwargs["order"],
                defaultKwargs["metric"],
            )

            # Generate input features kernels
            K_x_val = self._get_optimizer_kernel(
                X_val, None, self.kernel, self.sigma, self.nu, self.p
            )
            K_x_eval = self._get_optimizer_kernel(
                X_val, X_test, self.kernel, self.sigma, self.nu, self.p
            )

            K_val_composite = np.multiply(K_val, K_x_val)
            K_eval_composite = np.multiply(K_eval, K_x_eval)

            K_val_composite[np.diag_indices_from(K_val_composite)] += defaultKwargs[
                "reg"
            ]
            solved_coeffs = np.linalg.solve(K_val_composite, y_val)
            final_preds = np.dot(K_eval_composite, solved_coeffs)
            self.coeffs = solved_coeffs

        else:  # Default SGCT +-1 sub-model summation
            final_preds = np.zeros((X_test.shape[0]), dtype=float)
            count = 0
            for i in range(nfids):
                final_preds[:] += test_preds[:, count]
                count += 1
            for i in range(nfids - 1):
                final_preds -= test_preds[:, count]
                count += 1

        self.predict_time = time.time() - tstart

        if y_test is not None:
            self.mae = np.mean(np.abs(final_preds - y_test))
            self.rmse = np.sqrt(np.mean((final_preds - y_test) ** 2))

        return final_preds

import numpy as np
import time
import copy
from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor

# Import our custom classes and ultra-fast kernels
from .krr import (
    KRR,
    gaussian_kernel_symmetric, gaussian_kernel_asymmetric,
    matern_kernel_symmetric, matern_kernel_asymmetric,
    laplacian_kernel_symmetric, laplacian_kernel_asymmetric,
    wasserstein_kernel_symmetric, wasserstein_kernel_asymmetric
)
from .utils import property_differences

class ModelMFML:
    """Class to perform model difference MFML."""
    
    def __init__(self, reg: float = 1e-9, kernel: str = 'matern', sigma: float = 715.0,
                 nu: float = 1.5, p: float = 1.0, q: float = 1.0, p_bar: bool = False,
                 base_estimator: object = None):
        """
        Parameters
        ----------
        reg : float
            Regularization parameter for KRR. Defaults to 1e-9.
        kernel : str
            Kernel type ('matern', 'gaussian', 'laplacian', 'wasserstein', 'linear').
        sigma : float
            Kernel width parameter.
        nu : float
            Smoothness parameter for Matern kernel (0.5, 1.5, 2.5).
        p, q : float
            Parameters for Wasserstein kernel.
        p_bar : bool
            Enables or disables tqdm progress bar.
        base_estimator : object, optional
            A custom ML model instance to use. Default is the inbuilt KRR.
            Must have a `.fit(X, y)` or `.train(X, y)` method, and a `.predict(X)` method.
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
        Subsets the indexes to match n_trains while strictly retaining nested structure.
        If shuffle is True, it randomizes the selection deterministically based on the seed.
        If shuffle is False, it sequentially selects the first valid nested matches.
        """
        if n_trains is None:
            n_trains = np.asarray([self.indexes[i].shape[0] for i in range(self.indexes.shape[0])])
            
        nfids = self.indexes.shape[0]
        subset_index_array = np.zeros((nfids), dtype=object)
        
        # Get baseline indices. Shuffle at lowest fidelity if requested.
        baseline_indices = np.copy(self.indexes[0][:, 0])
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(baseline_indices)
        
        # Extract the required number of indices for each fidelity
        # following the universally consistent baseline order.
        for i in range(nfids):
            # Create a fast lookup for this fidelity's available mapping
            fid_map = {row[0]: row[1] for row in self.indexes[i]}
            
            patched_ind = []
            for b_idx in baseline_indices:
                if b_idx in fid_map:
                    patched_ind.append([b_idx, fid_map[b_idx]])
                    if len(patched_ind) == n_trains[i]:
                        break
                        
            subset_index_array[i] = np.asarray(patched_ind, dtype=int)
            
        return subset_index_array
    
    def y_train_breakup(self):
        n = self.indexes.shape[0]
        y_trains = np.zeros((2*n-1), dtype=object)
        count = 0
        
        for i in tqdm(range(n), desc='Extracting upper y_trains', leave=self.p_bar):
            ind_i = self.indexes[i][:, 1]
            y_trains[count] = np.copy(self.y_trains[i][ind_i])
            count += 1
            
        for i in tqdm(range(n-1), desc='Extracting lower y_trains', leave=self.p_bar):
            ind_i = self.indexes[i]
            ind_ip1 = self.indexes[i+1]
            c_i = []
            for row in ind_ip1:
                temp_i = np.where(ind_i[:, 0] == row[0])[0]
                if np.size(temp_i) != 0:
                    c_i.append(ind_i[temp_i[0], 1])
            y_trains[count] = np.copy(self.y_trains[i][np.asarray(c_i)])
            count += 1
            
        self.y_trains = y_trains
    
    def X_train_breakup(self):
        n = self.indexes.shape[0]
        X_trains = np.zeros((n), dtype=object)
        for i in tqdm(range(n), desc='Extracting X_trains', leave=self.p_bar):
            ind_i = self.indexes[i][:, 0]
            X_trains[i] = self.X_train_parent[ind_i]
        self.X_trains = np.copy(X_trains)
        
    def _get_optimizer_kernel(self, X1, X2, ktype, sigma, order_nu, metric_p):
        """Helper for the KRR/CompKRR optimizer kernels."""
        if ktype == 'gaussian':
            return gaussian_kernel_symmetric(X1, sigma) if X2 is None else gaussian_kernel_asymmetric(X1, X2, sigma)
        elif ktype == 'laplacian':
            return laplacian_kernel_symmetric(X1, sigma) if X2 is None else laplacian_kernel_asymmetric(X1, X2, sigma)
        elif ktype == 'matern':
            return matern_kernel_symmetric(X1, sigma, order_nu) if X2 is None else matern_kernel_asymmetric(X1, X2, sigma, order_nu)
        elif ktype == 'wasserstein':
            return wasserstein_kernel_symmetric(X1, sigma, order_nu, metric_p) if X2 is None else wasserstein_kernel_asymmetric(X1, X2, sigma, order_nu, metric_p)
        else:
            # Linear kernel fallback
            return np.dot(X1, X1.T) if X2 is None else np.dot(X2, X1.T)

    def _instantiate_and_train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Helper to cleanly instantiate a model and train it. Uses duck typing so it allows for any model architecture."""
        if self.base_estimator is None:
            model = KRR(kernel_type=self.kernel, sigma=self.sigma, nu=self.nu, p=self.p, q=self.q, reg=self.reg)
        else:
            model = copy.deepcopy(self.base_estimator)
            
        # Support both custom packages (.train) and scikit-learn standard (.fit)
        if hasattr(model, 'train'):
            model.train(X_train, y_train)
        elif hasattr(model, 'fit'):
            model.fit(X_train, y_train)
        else:
            raise AttributeError("The provided base_estimator must have either a '.train(X, y)' or '.fit(X, y)' method.")
            
        return model

    def train(self, X_train_parent: np.ndarray, file_paths: list = None, 
              y_trains: np.ndarray = None, indexes: np.ndarray = None, 
              shuffle: bool = False, n_trains: np.ndarray = None, seed: int = 0):
        tstart = time.time()
        self.X_train_parent = np.copy(X_train_parent)
        
        if y_trains is None and indexes is None:
            if file_paths is None:
                raise ValueError("Must provide either precomputed y_trains/indexes or file_paths.")
            self.y_trains, self.indexes = property_differences(file_paths)
        else:
            self.y_trains = y_trains
            self.indexes = indexes
        
        nfids = self.indexes.shape[0]
        # generate indexes/ shuffle as needed
        self.indexes = self._generate_nested_indexes(n_trains=n_trains, shuffle=shuffle, seed=seed)
        
        self.X_train_breakup() 
        self.y_train_breakup() 
        
        self.models = np.zeros((2 * nfids - 1), dtype=object)
        count = 0
        
        # Upper training
        for i in tqdm(range(nfids), desc='Training upper ML models...', leave=self.p_bar):
            self.models[count] = self._instantiate_and_train(self.X_trains[i], self.y_trains[count])
            count += 1
            
        # Lower training
        for i in tqdm(range(nfids - 1), desc='Training lower ML models', leave=self.p_bar):
            self.models[count] = self._instantiate_and_train(self.X_trains[i + 1], self.y_trains[count])
            count += 1
        
        self.train_time = time.time() - tstart

    def predict(self, X_test: np.ndarray, X_val: np.ndarray = None,
                y_test: np.ndarray = None, y_val: np.ndarray = None, 
                optimiser: str = 'default', **optargs):
        tstart = time.time()
        nfids = self.indexes.shape[0]
        
        test_preds = np.zeros((X_test.shape[0], 2 * nfids - 1), dtype=float)
        if y_val is not None:
            val_preds = np.zeros((X_val.shape[0], 2 * nfids - 1), dtype=float)
        
        count = 0
        # Upper triangle preds
        for i in tqdm(range(nfids), desc='Upper MFML predictions', leave=self.p_bar):
            if y_val is not None:
                val_preds[:, count] = self.models[count].predict(X_val)
            test_preds[:, count] = self.models[count].predict(X_test)
            count += 1
            
        # Lower triangle preds
        for i in tqdm(range(nfids - 1), desc='Lower MFML predictions', leave=self.p_bar):
            if y_val is not None:
                val_preds[:, count] = self.models[count].predict(X_val)
            test_preds[:, count] = self.models[count].predict(X_test)
            count += 1
        
        # optimzers for o-MFML and related
        if optimiser == 'OLS':
            defaultKwargs = {'copy_X': True, 'fit_intercept': False}
            defaultKwargs.update(**optargs)
            regressor = LinearRegression(**defaultKwargs)
            regressor.fit(val_preds, y_val)
            final_preds = regressor.predict(test_preds)
            self.LCCoptimizer = regressor
        
        elif optimiser == 'LRR':
            defaultKwargs = {'alpha': 1e-9, 'fit_intercept': False, 'copy_X': True}
            defaultKwargs.update(**optargs)
            regressor = Ridge(**defaultKwargs)
            regressor.fit(val_preds, y_val)
            final_preds = regressor.predict(test_preds)
            self.LCCoptimizer = regressor
        
        elif optimiser == 'LASSO':
            defaultKwargs = {'alpha': 1.0, 'fit_intercept': False, 'max_iter': 1000}
            defaultKwargs.update(**optargs)
            regressor = Lasso(**defaultKwargs)
            regressor.fit(val_preds, y_val)
            final_preds = regressor.predict(test_preds)
            self.LCCoptimizer = regressor
        
        elif optimiser == 'MLPR':
            defaultKwargs = {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam'}
            defaultKwargs.update(**optargs)
            MLPR = MLPRegressor(**defaultKwargs)
            MLPR.fit(val_preds, y_val)
            final_preds = MLPR.predict(test_preds)
            self.LCCoptimizer = MLPR
        
        elif optimiser == 'KRR':
            defaultKwargs = {'sigma': 700.0, 'reg': 1e-9, 'kernel_type': 'gaussian', 'order': 1.5, 'metric': 1.0}
            defaultKwargs.update(**optargs)
            
            K_val = self._get_optimizer_kernel(val_preds, None, defaultKwargs['kernel_type'], defaultKwargs['sigma'], defaultKwargs['order'], defaultKwargs['metric'])
            K_eval = self._get_optimizer_kernel(val_preds, test_preds, defaultKwargs['kernel_type'], defaultKwargs['sigma'], defaultKwargs['order'], defaultKwargs['metric'])

            K_val[np.diag_indices_from(K_val)] += defaultKwargs['reg']
            opt_alpha = np.linalg.solve(K_val, y_val)
            final_preds = np.dot(K_eval, opt_alpha)
            self.coeffs = opt_alpha

        elif optimiser == 'CompKRR':
            defaultKwargs = {'sigma': 700.0, 'reg': 1e-9, 'kernel_type': 'gaussian', 'order': 1.5, 'metric': 1.0}
            defaultKwargs.update(**optargs)
            
            K_val = self._get_optimizer_kernel(val_preds, None, defaultKwargs['kernel_type'], defaultKwargs['sigma'], defaultKwargs['order'], defaultKwargs['metric'])
            K_eval = self._get_optimizer_kernel(val_preds, test_preds, defaultKwargs['kernel_type'], defaultKwargs['sigma'], defaultKwargs['order'], defaultKwargs['metric'])

            # Generate input features kernels
            K_x_val = self._get_optimizer_kernel(X_val, None, self.kernel, self.sigma, self.nu, self.p)
            K_x_eval = self._get_optimizer_kernel(X_val, X_test, self.kernel, self.sigma, self.nu, self.p)
            
            K_val_composite = np.multiply(K_val, K_x_val)
            K_eval_composite = np.multiply(K_eval, K_x_eval)
            
            K_val_composite[np.diag_indices_from(K_val_composite)] += defaultKwargs['reg']
            solved_coeffs = np.linalg.solve(K_val_composite, y_val)
            final_preds = np.dot(K_eval_composite, solved_coeffs)
            self.coeffs = solved_coeffs
            
        else: # Default SGCT logic
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
            self.rmse = np.sqrt(np.mean((final_preds - y_test)**2))
        
        return final_preds

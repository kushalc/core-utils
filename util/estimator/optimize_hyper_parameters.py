#! /usr/bin/env python3

import json
import logging
import os

import cloudpickle
import numpy as np
import pandas as pd
from scipy import optimize

# from skopt import BayesSearchCV, space
from util.serialization import load_class
from util.shared import parse_args

RESULTS_DF = pd.DataFrame(columns=["score", "params", "accepted"])
def _build_scipy_optimizable(estimator, step_size, seed, subset=[]):
    bounds_df = _build_scipy_bounds(estimator)
    params_df = pd.DataFrame(estimator.get_params().items(), columns=["name", "value"]) \
                  .set_index("name") \
                  .reindex(bounds_df.index)
    if subset is not None and len(subset):
        params_df = params_df.reindex(subset)
        bounds_df = bounds_df.reindex(subset)
    def __optimizable(params, X, y_gold):
        # FIXME: The first call to this method returns results consistent w/confusion matrix,
        # but subsequent calls diverge. In fact, the 2nd score returned by this method is invariably
        # incorrect and the same as the 1st. Why?
        #
        # FIXME: For reasons I don't understand, this doesn't work:
        # current = clone(estimator).set_params(dict(zip(params_df.index, params)))
        #
        # But the following does. Does this mean we've got something wrong w/caching and/or clone()?
        current = estimator.get(force=True, use_s3=False, use_memory=False, **dict(zip(params_df.index, params)))
        y_pred = current._cv_predict(current, X, y_gold, random_state=seed)
        score = -current._metric_score(y_gold, y_pred)
        return score

    def __callback(params, energy, accepted, **kwargs):
        score_max = RESULTS_DF["score"].max()
        RESULTS_DF.loc[len(RESULTS_DF)] = [-energy, dict(zip(params_df.index, params)), accepted]
        if RESULTS_DF.iloc[-1]["score"] > score_max:
            _update_estimator_params(estimator)

    from scipy.optimize._basinhopping import RandomDisplacement
    class BoundedRandomDisplacement(RandomDisplacement):
        def __displace(self, x):
            # approximates truncated normal distribution with mode at x and stddev of 3*self.stepsize.
            range_s = (bounds_df["max"] - bounds_df["min"]).values
            if len(range_s) > 1:
                for _ in range(self.random_state.randint(len(range_s)-1)):  # mask out up tp N-1 displacements
                    range_s[self.random_state.randint(len(range_s))] = 0
            delta_s = self.random_state.triangular(-self.stepsize, 0, self.stepsize, np.shape(range_s))
            return x + range_s * delta_s

        def __within_bounds(self, x_p):
            return (bounds_df["min"].values <= x_p) & (x_p <= bounds_df["max"].values)

        def __call__(self, x):
            x_p = self.__displace(x)
            while not self.__within_bounds(x_p).all():
                x_p = self.__displace(x)

            logging.info("Generated new Metropolis proposal:\n%s", pd.DataFrame({ "original": x, "proposal": x_p },
                                                                                index=params_df.index).to_string())
            return x_p

    initial = params_df["value"].values
    return __optimizable, __callback, initial, params_df.index, BoundedRandomDisplacement(random_state=np.random.RandomState(seed),
                                                                                          stepsize=step_size)

def _build_scipy_bounds(estimator):
    landscape = pd.DataFrame(columns=["min", "max"])

    constraints = json.load(open(estimator._params_path("constraints")))
    for name, options in constraints.items():
        landscape.loc[name, "min"] = options["nargs"][0]
        landscape.loc[name, "max"] = options["nargs"][1]

    return landscape

def _build_skopt_landscape(estimator):
    return { name: getattr(space, options["type"])(*options.get("nargs", []), **options.get("kwargs", {}))
             for name, options in _build_scipy_bounds(estimator).iterrows() }

def _update_estimator_params(estimator, iloc=0):
    sorted_df = RESULTS_DF.sort_values("score", ascending=False)
    params = dict(estimator._params_json())
    params.update(sorted_df.iloc[iloc]["params"])
    for k in list(k for k in params if k.startswith("__")):
        params.pop(k, None)
    estimator.set_params(**params)

    params["__score"] = sorted_df.iloc[iloc]["score"]
    if pd.isnull(params["__score"]):
        params["__score"] = estimator._cv_scorer(estimator, *estimator.goldset())

    kwargs = dict(indent=2, separators=[',', ': '], sort_keys=True)
    json.dump(params, open(estimator._params_path(), "w"), **kwargs)
    logging.info("Updated %s params JSON: %s: %s", estimator.name(), estimator._params_path(), json.dumps(params, **kwargs))
    return sorted_df

if __name__ == "__main__":
    args = parse_args("Optimize hyper-parameters for any BaseEstimator-based model.", [
        {"name_or_flags": "--no-force", "dest": "force", "action": "store_false", "help": "don't rebuild model from scratch" },
        {"name_or_flags": "--iterations", "type": int, "default": 100, "help": "how many iterations of hyper-parameter optimization" },
        {"name_or_flags": "--step-size", "type": float, "default": 0.125, "help": "average per-dimension percentage of distance for next proposal" },
        {"name_or_flags": "--temperature", "type": float, "default": 1.000, "help": "average difference between scores for successive iterations" },
        {"name_or_flags": "model_cls", "type": load_class, "help": "load this class" },
        {"name_or_flags": "optimizable", "nargs":"*", "default": [], "help": "parameters to optimize" },
    ])

    estimator = args.model_cls.get(force=args.force, use_s3=False)  # disabling S3 upload for big models
    optimizable, callback, initial, args.optimizable, stepper = _build_scipy_optimizable(estimator, args.step_size, args.seed, subset=args.optimizable)

    # differential evolution — callback doesn't return energy
    # popsize = 5
    # initial = np.tile(initial.reshape(1, -1), (popsize, 1)) + np.random.uniform(-0.125, 0.125, (popsize, 4))
    # result = optimize.differential_evolution(optimizable, maxiter=args.iterations, args=estimator.goldset(),
    #                                          bounds=_build_scipy_bounds(estimator).values, popsize=popsize,
    #                                          init=initial, polish=False, callback=callback, disp=True)

    # basinhopping — can't get local optimization to work
    def __dummy_minimize(fun, x0, *nargs, **kwargs):
        return optimize.OptimizeResult(x=x0, fun=fun(x0, *kwargs["args"]), success=True)
    callback(initial, optimizable(initial, *estimator.goldset()), False)  # doesn't give us initial, so initialize externally w/known baseline.
    # lbfgs_options = dict(disp=101, maxiter=5, maxls=5, maxfun=25, ftol=5e-3, gtol=1e-3, eps=1e-3)
    minimizer_kwargs = dict(method=__dummy_minimize, args=estimator.goldset(),
                            bounds=optimize.Bounds(lb=_build_scipy_bounds(estimator).loc[args.optimizable, "min"].values,
                                                   ub=_build_scipy_bounds(estimator).loc[args.optimizable, "max"].values))
    result = optimize.basinhopping(optimizable, x0=initial, callback=callback, disp=True, T=args.temperature,
                                   interval=5, stepsize=args.step_size, niter=args.iterations,
                                   take_step=stepper, minimizer_kwargs=minimizer_kwargs)

    # gaussian process — doesn't seem to converge
    # optimized = BayesSearchCV(estimator, _build_skopt_landscape(estimator), n_iter=args.iterations,
    #                           scoring=estimator.scorer, cv=estimator._cv_splitter(), error_score=np.nan)
    # optimized.fit(*estimator.goldset())
    #
    # results_df = pd.DataFrame(optimized.cv_results_)
    # results_df["lb_test_score"] = results_df["mean_test_score"] - results_df["std_test_score"]
    # results_df.sort_values("lb_test_score", ascending=False, inplace=True)
    # results_df.drop(columns=[c for c in results_df if c.startswith("split") or c in ["std_fit_time", "std_score_time", "params", "rank_test_score", "lb_test_score"]], inplace=True)
    #
    # params = optimized.best_params_
    # params["__score"] = optimized.best_score_

    results_df = _update_estimator_params(estimator)
    def _output(ext):
        return os.path.join(args.output, "{:%H-%M}.{:}".format(args.started_at, ext))
    logging.info("Completed hyper-parameter optimization:\n%s", results_df.to_string())
    cloudpickle.dump(results_df, open(_output("cloudpickle"), "wb"))
    import pdb; pdb.set_trace()

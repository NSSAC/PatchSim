#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" PatchSim v1.2
Created and maintained by: Srini (srini@virginia.edu)
Date last modified: 6 Aug 2019
"""
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def read_config(config_file):
    """Read configuration.

    Configuration files contain one key=value pair per line.
    The following is an example of the contents of a config file::

        PatchFile=test_pop.txt
        NetworkFile=test_net.txt
        NetworkType=Static

        ExposureRate=0.65
        InfectionRate=0.67
        RecoveryRate=0.4
        ScalingFactor=1

        SeedFile=test_seed.txt
        VaxFile=test_vax.txt
        VaxDelay=4
        VaxEfficacy=0.5

        StartDate=1
        Duration=30

        LoadState=False
        SaveState=True
        SaveFile=checkpoint1.npy

        OutputFile=test1.out
        OutputFormat=Whole
        LogFile=test1.log

    Parameters
    ----------
    config_file : str
        Path to the configuration file.

    Returns
    -------
    dict (str -> str)
        The configuration key value pairs.
    """
    config_df = pd.read_csv(config_file, delimiter="=", names=["key", "val"])
    configs = dict(zip(config_df.key, config_df.val))
    configs.setdefault("Model", "Mobility")
    return configs


def load_patch(configs):
    """Load the patch file.

    A patch file contains the population size of a patch.
    The file has two space separated columns.
    Following is an example of a patch file::

        A 10000
        B 10000
        C 10000

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
        Must contain the "PatchFile" pointing to location of patch file.

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with following columns.
        id : dtype=str
        pops : dtype=int
    """
    patch_df = pd.read_csv(
        configs["PatchFile"],
        names=["id", "pops"],
        delimiter=" ",
        dtype={"id": str, "pops": int},
    )
    patch_df.sort_values("id", inplace=True)

    logger.info("Loaded patch attributes")
    return patch_df


def load_param_file(configs):
    """Load the parameter file.

    A parameter file contains one row per patch.
    Each row must have two or more columns.
    Following is an example of a paremter file::

        B 0 0 0.54 0.54 0.54 0.54 0 0 0 0
        A 0.72

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
        Must contain the "ParamFile" pointing to location of parameter file.
    patch_df : pd.DataFrame
        A pandas dataframe with following columns.
        id : dtype=str
        pops : dtype=int

    Returns
    -------
    pd.DataFrame
         A pandas dataframe with one column per patch
         The column names are IDs of the patches.
    """
    param_df = pd.read_csv(
        configs["ParamFile"], delimiter=" ", dtype={0: str}, header=None
    )
    param_df = param_df.set_index(0)
    param_df = param_df.fillna(method="ffill", axis=1)
    param_df = param_df.T

    return param_df


def load_params(configs, patch_df):
    """Load the simulation parameters.

    Parameters
    ----------
    configs : dict (str -> str)
        The configuration key value pairs.
    patch_df : pd.DataFrame
        A pandas dataframe with following columns.
        id : dtype=str
        pops : dtype=int

    Returns
    -------
    dict (str -> float or ndarray)
        A dictionary of model parameters.
        The "beta" parameter is a matrix (NumPatchesxNumTimesteps).
    """
    params = {}
    params["T"] = int(configs["Duration"])

    beta = float(configs.get("ExposureRate", 0.0))
    params["beta"] = np.full((len(patch_df), params["T"]), beta)
    params["alpha"] = float(configs.get("InfectionRate", 0.0))
    params["gamma"] = float(configs.get("RecoveryRate", 0.0))
    logger.info(
        "Parameter: alpha=%e, beta=%e, gamma=%e", params["alpha"], beta, params["gamma"]
    )

    if "ParamFile" in configs:
        param_df = load_param_file(configs)
        for i, id_ in enumerate(patch_df["id"]):
            if id_ in param_df.columns:
                xs = param_df[id_]
                params["beta"][i, 0 : len(xs)] = xs
        logger.info("Loaded disease parameters from ParamFile")
    else:
        logger.info("No ParamFile loaded")

    ### Optional parameters
    params["scaling"] = float(configs.get("ScalingFactor", 1.0))
    params["vaxeff"] = float(configs.get("VaxEfficacy", 1.0))
    params["delta"] = float(configs.get("WaningRate", 0.0))
    params["kappa"] = 1 - float(configs.get("AsymptomaticReduction", 0.0))
    params["symprob"] = float(configs.get("SymptomaticProbability", 1.0))
    params["epsilon"] = float(configs.get("PresymptomaticReduction", 1.0))

    if params["delta"]:
        logger.info("Found WaningRate. Running SEIRS model.")

    return params


def load_seed(configs, params, patch_df):
    """Load the disease seeding schedule file.

    A seed file contains the disease seeding schedule.
    Following is an example of the contents of a seed file::

        0 A 20
        0 B 20
        1 C 20
        2 C 30

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
    params: dict (str -> float or ndarray)
        A dictionary of model parameters.
    patch_df : pd.DataFrame
        A pandas dataframe with following columns.
        id : dtype=str
        pops : dtype=int

    Returns
    -------
    np.ndarray
        A seeding schedule matrix (NumTimstepsxNumPatches)
    """
    if "SeedFile" not in configs:
        logger.info("Continuing without seeding")
        return np.zeros((params["T"], len(patch_df)))

    seed_df = pd.read_csv(
        configs["SeedFile"],
        delimiter=" ",
        names=["Day", "Id", "Count"],
        dtype={"Id": str},
    )

    seed_mat = np.zeros((params["T"], len(patch_df)))
    patch_idx = {id_: i for i, id_ in enumerate(patch_df["id"])}
    for id_, day, count in zip(seed_df["Id"], seed_df["Day"], seed_df["Count"]):
        idx = patch_idx[id_]
        seed_mat[day, idx] = count

    logger.info("Loaded seeding schedule")
    return seed_mat


def load_vax(configs, params, patch_df):
    """Load the vaccination schedule file.

    A vax file contains the vaccination schedule.
    Following is an example of the contents of the vax file::

        0 A 10
        2 B 10
        5 C 10

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
    params: dict (str -> float or ndarray)
        A dictionary of model parameters.
    patch_df : pd.DataFrame
        A pandas dataframe with following columns.
        id : dtype=str
        pops : dtype=int

    Returns
    -------
    np.ndarray
        A vaccination schedule matrix (NumTimstepsxNumPatches)
    """
    vax_mat = np.zeros((params["T"], len(patch_df)), dtype=int)

    if "VaxFile" not in configs:
        return vax_mat

    vax_df = pd.read_csv(
        configs["VaxFile"],
        delimiter=" ",
        names=["Day", "Id", "Count"],
        dtype={"Id": str, "Count": int},
    )
    vax_delay = int(configs.get("VaxDelay", 0))

    patch_idx = {id_: i for i, id_ in enumerate(patch_df["id"])}
    for id_, day, count in zip(vax_df["Id"], vax_df["Day"], vax_df["Count"]):
        idx = patch_idx[id_]
        day = day + vax_delay
        vax_mat[day, idx] = count

    return vax_mat


def load_Theta(configs, patch_df):
    theta_df = pd.read_csv(
        configs["NetworkFile"],
        names=["src_Id", "dest_Id", "theta_index", "flow"],
        delimiter=" ",
        dtype={"src_Id": str, "dest_Id": str},
    )

    if (configs["NetworkType"] == "Static") & (len(theta_df.theta_index.unique()) != 1):
        logger.info("Theta indices mismatch. Ensure NetworkType=Static.")
    if (configs["NetworkType"] == "Weekly") & (
        len(theta_df.theta_index.unique()) != 53
    ):
        logger.info("Theta indices mismatch. Ensure NetworkType=Weekly.")
    if (configs["NetworkType"] == "Monthly") & (
        len(theta_df.theta_index.unique()) != 12
    ):
        logger.info("Theta indices mismatch. Ensure NetworkType=Monthly.")

    patch_idx = dict(zip(patch_df.id.values, range(len(patch_df))))
    try:
        theta_df["src_Id_int"] = theta_df.src_Id.apply(lambda x: patch_idx[x])
        theta_df["dest_Id_int"] = theta_df.dest_Id.apply(lambda x: patch_idx[x])
    except:
        logger.info(
            "Ignoring flow entries for missing patches. Ensure all patches listed in PatchFile."
        )

    Theta_indices = theta_df.theta_index.unique()
    Theta = np.ndarray((len(Theta_indices), len(patch_df), len(patch_df)))

    for k in Theta_indices:
        theta_df_k = theta_df[theta_df.theta_index == k]
        theta_df_k = theta_df_k.pivot(
            index="src_Id_int", columns="dest_Id_int", values="flow"
        ).fillna(0)
        theta_df_k = theta_df_k.reindex(
            index=range(len(patch_df)), columns=range(len(patch_df))
        ).fillna(0)
        Theta[int(k)] = theta_df_k.values

    logger.info("Loaded temporal travel matrix")
    return Theta


def patchsim_step(State_Array, patch_df, configs, params, theta, seeds, vaxs, t, stoch):
    S, E, I, R, V, new_inf = State_Array  ## Aliases for the State Array

    ## seeding for day t (seeding implies S->I)
    actual_seed = np.minimum(seeds[t], S[t])
    S[t] = S[t] - actual_seed
    I[t] = I[t] + actual_seed

    if stoch:
        ## vaccination for day t
        max_SV = np.minimum(vaxs[t], S[t])
        actual_SV = np.random.binomial(max_SV.astype(int), params["vaxeff"])
        S[t] = S[t] - actual_SV
        V[t] = V[t] + actual_SV

        ## Computing force of infection
        ## Modify this to do travel network sampling only once and use it for the entire simulation.
        ## Or even skip network sampling altogether, and model only disease progression stochasticity

        N = patch_df.pops.values
        S_edge = np.concatenate(
            [
                np.random.multinomial(
                    S[t][x], theta[x] / (theta[x].sum() + 10 ** -12)
                ).reshape(1, len(N))
                for x in range(len(N))
            ],
            axis=0,
        )
        E_edge = np.concatenate(
            [
                np.random.multinomial(
                    E[t][x], theta[x] / (theta[x].sum() + 10 ** -12)
                ).reshape(1, len(N))
                for x in range(len(N))
            ],
            axis=0,
        )
        I_edge = np.concatenate(
            [
                np.random.multinomial(
                    I[t][x], theta[x] / (theta[x].sum() + 10 ** -12)
                ).reshape(1, len(N))
                for x in range(len(N))
            ],
            axis=0,
        )
        R_edge = np.concatenate(
            [
                np.random.multinomial(
                    R[t][x], theta[x] / (theta[x].sum() + 10 ** -12)
                ).reshape(1, len(N))
                for x in range(len(N))
            ],
            axis=0,
        )
        V_edge = np.concatenate(
            [
                np.random.multinomial(
                    V[t][x], theta[x] / (theta[x].sum() + 10 ** -12)
                ).reshape(1, len(N))
                for x in range(len(N))
            ],
            axis=0,
        )
        N_edge = S_edge + E_edge + I_edge + R_edge + V_edge

        N_eff = N_edge.sum(axis=0)
        I_eff = I_edge.sum(axis=0)
        beta_j_eff = np.nan_to_num(params["beta"][:, t] * (I_eff / N_eff))

        actual_SE = np.concatenate(
            [
                np.random.binomial(S_edge[:, x], beta_j_eff[x]).reshape(len(N), 1)
                for x in range(len(N))
            ],
            axis=1,
        ).sum(axis=1)
        actual_EI = np.random.binomial(E[t], params["alpha"])
        actual_IR = np.random.binomial(I[t], params["gamma"])
        actual_RS = np.random.binomial(R[t], params["delta"])

        ### Update to include presymptomatic and asymptomatic terms
        S[t + 1] = S[t] - actual_SE + actual_RS
        E[t + 1] = E[t] + actual_SE - actual_EI
        I[t + 1] = I[t] + actual_EI - actual_IR
        R[t + 1] = R[t] + actual_IR - actual_RS
        V[t + 1] = V[t]

    else:
        ## vaccination for day t
        actual_vax = np.minimum(vaxs[t] * params["vaxeff"], S[t])
        S[t] = S[t] - actual_vax
        V[t] = V[t] + actual_vax

        N = patch_df.pops.values

        ## Computing force of infection

        if configs["Model"] == "Mobility":
            N_eff = theta.T.dot(N)
            I_eff = theta.T.dot(I[t])
            E_eff = theta.T.dot(E[t])
            beta_j_eff = np.nan_to_num(
                np.multiply(
                    np.divide(I_eff, N_eff),
                    params["beta"][:, t]
                    * (
                        (1 - params["kappa"]) * (1 - params["symprob"])
                        + params["symprob"]
                    ),
                )
            )  ## force of infection from symp/asymptomatic individuals
            E_beta_j_eff = np.nan_to_num(
                np.multiply(
                    np.divide(E_eff, N_eff),
                    params["beta"][:, t] * (1 - params["epsilon"]),
                )
            )  ##force of infection from presymptomatic individuals
            inf_force = theta.dot(beta_j_eff + E_beta_j_eff)

        elif configs["Model"] == "Force":
            beta_j_eff = np.nan_to_num(
                np.multiply(np.divide(I[t], N), params["beta"][:, t])
            )
            # print(beta_j_eff)
            inf_force = theta.T.dot(beta_j_eff)
            # print(inf_force)

        ## New exposures during day t
        new_inf[t] = np.minimum(
            np.multiply(inf_force, S[t]), S[t]
        )  ## Maximum number of new infections at time t is S[t]
        # print(new_inf)
        ### Update to include presymptomatic and asymptomatic terms
        S[t + 1] = S[t] - new_inf[t] + np.multiply(params["delta"], R[t])
        E[t + 1] = new_inf[t] + np.multiply(1 - params["alpha"], E[t])
        I[t + 1] = np.multiply(params["alpha"], E[t]) + np.multiply(
            1 - params["gamma"], I[t]
        )
        R[t + 1] = np.multiply(params["gamma"], I[t]) + np.multiply(
            1 - params["delta"], R[t]
        )
        V[t + 1] = V[t]


def epicurves_todf(configs, params, patch_df, State_Array):
    S, E, I, R, V, new_inf = State_Array  ## Aliases for the State Array

    out_df = pd.DataFrame(
        index=patch_df.id.values,
        columns=range(int(configs["Duration"])),
        data=new_inf[:-1, :].T,
    )
    out_df = out_df * float(params["scaling"])
    if configs["OutputFormat"] == "Whole":
        out_df = out_df.round().astype(int)

    return out_df


def write_epicurves(configs, params, patch_df, State_Array, write_epi, return_epi):

    out_df = epicurves_todf(configs, params, patch_df, State_Array)

    if (write_epi == False) & (return_epi == False):
        return out_df.sum().sum()
    else:
        if write_epi == True:
            out_df.to_csv(configs["OutputFile"], header=None, sep=" ")

        if return_epi == True:
            return out_df

        return


def run_disease_simulation(
    configs,
    patch_df=None,
    params=None,
    Theta=None,
    seeds=None,
    vaxs=None,
    return_epi=False,
    write_epi=False,
    return_full=False,
):
    try:
        handler = logging.FileHandler(configs["LogFile"], mode="w")
        for hdlr in logger.handlers[:]:  # remove the existing file handlers
            if isinstance(hdlr, logger.FileHander):
                logger.removeHandler(hdlr)

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
    except:
        handler = logging.NullHandler()
        logger.addHandler(handler)

    logger.info("Starting PatchSim")
    start = time.time()

    if configs["Model"] not in ["Mobility", "Force"]:
        logger.info("Invalid Model for PatchSim")
        logger.removeHandler(handler)
        return
    else:
        logger.info("Operating PatchSim under {} Model".format(configs["Model"]))

    if patch_df is None:
        patch_df = load_patch(configs)

    if params is None:
        params = load_params(configs, patch_df)

    if Theta is None:
        Theta = load_Theta(configs, patch_df)

    if seeds is None:
        seeds = load_seed(configs, params, patch_df)

    if vaxs is None:
        vaxs = load_vax(configs, params, patch_df)

    logger.info("Initializing simulation run...")

    if "RandomSeed" in configs.keys():
        np.random.seed(int(configs["RandomSeed"]))
        stoch = True
        logger.info("Found RandomSeed. Running in stochastic mode...")
    else:
        stoch = False
        logger.info("No RandomSeed found. Running in deterministic mode...")

    dim = 5 + 1  ##Number of states (SEIRV) + One for tracking new infections
    if stoch:
        State_Array = np.ndarray((dim, params["T"] + 1, len(patch_df))).astype(int)
    else:
        State_Array = np.ndarray((dim, params["T"] + 1, len(patch_df)))

    State_Array.fill(0)
    S, E, I, R, V, new_inf = State_Array  ## Aliases for the State Array

    if configs["LoadState"] == "True":
        State_Array[:, 0, :] = np.load(configs["LoadFile"])
    else:
        S[0, :] = patch_df.pops.values

    ref = datetime.strptime("Jan 1 2017", "%b %d %Y")  ##is a Sunday
    for t in range(params["T"]):
        curr_date = ref + timedelta(days=t + int(configs["StartDate"]))
        curr_week = int(curr_date.strftime("%U"))
        curr_month = int(curr_date.strftime("%m"))

        if configs["NetworkType"] == "Static":
            patchsim_step(
                State_Array, patch_df, configs, params, Theta[0], seeds, vaxs, t, stoch
            )

        if configs["NetworkType"] == "Weekly":
            patchsim_step(
                State_Array,
                patch_df,
                configs,
                params,
                Theta[curr_week - 1],
                seeds,
                vaxs,
                t,
                stoch,
            )

        if configs["NetworkType"] == "Monthly":
            patchsim_step(
                State_Array,
                patch_df,
                configs,
                params,
                Theta[curr_month - 1],
                seeds,
                vaxs,
                t,
                stoch,
            )

    if configs["SaveState"] == "True":
        logger.info("Saving StateArray to File")
        np.save(configs["SaveFile"], State_Array[:, -1, :])

    elapsed = time.time() - start
    logger.info("Simulation complete. Time elapsed: {} seconds.".format(elapsed))
    logger.removeHandler(handler)

    #     if (return_full==True): ##Use for debugging
    #         return State_Array
    return write_epicurves(
        configs, params, patch_df, State_Array, write_epi, return_epi
    )

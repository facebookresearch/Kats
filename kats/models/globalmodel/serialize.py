# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from multiprocessing import cpu_count
from typing import Any, Dict, Union

from kats.models.globalmodel.ensemble import GMEnsemble
from kats.models.globalmodel.model import GMModel
from kats.models.globalmodel.utils import GMParam, gmparam_from_string
from torch import Tensor


def _gmmodel_nn_to_dict(gm: GMModel) -> Dict[str, Any]:
    ans = {}
    if gm.params.model_type == "rnn":
        ans["decoder"] = None
        ans["encoder"] = None
        state_dict = gm.rnn.state_dict()
        ans["rnn"] = {t: state_dict[t].numpy().tolist() for t in state_dict}
    else:
        ans["rnn"] = None
        for name in ["decoder", "encoder"]:
            state_dict = getattr(gm, name).state_dict()
            ans[name] = {t: state_dict[t].numpy().tolist() for t in state_dict}
    return ans


def _dict_to_gmmodel_nn(gmparam: GMParam, nn_dict: Dict[str, Any]) -> GMModel:
    gm = GMModel(gmparam)
    gm._initiate_nn()

    if gmparam.model_type == "rnn":
        state_dict = {t: Tensor(nn_dict["rnn"][t]) for t in nn_dict["rnn"]}
        gm.rnn.load_state_dict(state_dict)
    else:
        for name in ["decoder", "encoder"]:
            state_dict = {t: Tensor(nn_dict[name][t]) for t in nn_dict[name]}
            getattr(gm, name).load_state_dict(state_dict)
    return gm


def global_model_to_json(gme: Union[GMModel, GMEnsemble]) -> str:
    if isinstance(gme, GMEnsemble):
        # recording basic params of gme
        ans = {
            t: getattr(gme, t)
            for t in [
                "splits",
                "overlap",
                "replicate",
                "model_num",
                "multi",
                "max_core",
                "gm_info",
                "ensemble_type",
            ]
        }

        ans["gmparam"] = gme.params.to_string()
        ans["gm_models"] = [_gmmodel_nn_to_dict(t) for t in gme.gm_models]
    elif isinstance(gme, GMModel):
        ans = {}
        ans["gmparam"] = gme.params.to_string()
        ans.update(_gmmodel_nn_to_dict(gme))
    else:
        msg = f"Wrong input data type. We expect to recive GMModel or GMEnsemble but receive {type(gme)}."
        logging.error(msg)
        raise ValueError(msg)

    return json.dumps(ans)


def load_global_model_from_json(json_str: str) -> Union[GMModel, GMEnsemble]:

    param_dict = json.loads(json_str)

    # string for GMEnsemble
    if set(param_dict.keys()) == {
        "splits",
        "overlap",
        "replicate",
        "model_num",
        "multi",
        "max_core",
        "gm_info",
        "gm_models",
        "gmparam",
        "ensemble_type",
    }:
        gmparam = gmparam_from_string(param_dict["gmparam"])

        ans = GMEnsemble(
            gmparam=gmparam,
            splits=param_dict["splits"],
            overlap=param_dict["overlap"],
            multi=param_dict["multi"],
            replicate=param_dict["replicate"],
            max_core=min(cpu_count(), param_dict["max_core"]),
            ensemble_type=param_dict["ensemble_type"],
        )
        ans.gm_info = param_dict["gm_info"]
        ans.gm_models = [
            _dict_to_gmmodel_nn(gmparam, t) for t in param_dict["gm_models"]
        ]

    # string for GMModel
    elif set(param_dict.keys()) == {"gmparam", "rnn", "decoder", "encoder"}:
        gmparam = gmparam_from_string(param_dict["gmparam"])
        ans = _dict_to_gmmodel_nn(gmparam, param_dict)

    else:
        msg = "Fail to load global model."
        logging.error(msg)
        raise ValueError(msg)
    return ans

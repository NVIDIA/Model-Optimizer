#!/bin/bash

if [[ -z "${DOCKER_BUILD:-}" ]]; then
    echo "Running outside Docker build"

    # the user might have provided a script to provide cache dirs
    SOURCE_FILE=""

    parse_args() {
	while [[ $# -gt 0 ]]; do
            case "$1" in
		--source)
                    if [[ -n "$SOURCE_FILE" ]]; then
			echo "Error: --source may only be specified once" >&2
			return 2
                    fi

                    if [[ $# -lt 2 ]]; then
			echo "Error: --source requires a file" >&2
			return 2
                    fi

                    SOURCE_FILE=$2
                    shift 2
                    ;;
		--venv)
                    if [[ -n "$VENV_OPTION" ]]; then
			echo "Error: --venv may only be specified once" >&2
			return 2
                    fi

                    if [[ $# -lt 2 || "$2" == --* ]]; then
			echo "Error: --venv requires a path" >&2
			return 2
                    fi

                    VENV_OPTION=$2
                    shift 2
                    ;;

		*)
                    echo "Error: unknown option: $1" >&2
                    return 2
                    ;;
            esac
	done
    }

    load_source_file() {
	[[ -z "$SOURCE_FILE" ]] && return 0

	if [[ ! -f "$SOURCE_FILE" ]]; then
            echo "Error: file does not exist: $SOURCE_FILE" >&2
            return 2
	fi

	if [[ "$SOURCE_FILE" != *.sh ]]; then
            echo "Error: source file must end in .sh: $SOURCE_FILE" >&2
            return 2
	fi

	# shellcheck source=/dev/null
	source "$SOURCE_FILE"
    }

    configure_environment() {
	if [[ -n "$VENV_OPTION" ]]; then
            VIRTUAL_ENV=$VENV_OPTION
	else
            VIRTUAL_ENV=${VIRTUAL_ENV:-.venv}
	fi

	export VIRTUAL_ENV
	echo "Using VIRTUAL_ENV=$VIRTUAL_ENV"
    }

    parse_args "$@" || exit $?
    load_source_file || exit $?
    configure_environment
    
fi

if [[ ! -d "$VIRTUAL_ENV" ]]; then
    uv venv $VIRTUAL_ENV --python 3.12 --seed --managed-python
fi

echo "Sourcing virtual env at $VIRTUAL_ENV"
source $VIRTUAL_ENV/bin/activate


uv pip install -e .[dev]
uv pip install -r examples/puzzletron/requirements.txt
# export VLLM_USE_PRECOMPILED=1
uv pip install git+https://github.com/grzegorz-k-karch/vllm.git@feature/add_anymodel_to_vllm_wip --torch-backend=auto

uv pip install git+https://github.com/Separius/Automodel.git@puzzletron
uv pip install aiperf

uv pip install --upgrade --force-reinstall --no-cache-dir \
       torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
       --index-url https://download.pytorch.org/whl/cu130

uv pip install --no-build-isolation --no-deps --no-binary=mamba-ssm mamba-ssm==2.3.2.post1

.venv/bin/python -m pip uninstall -y causal-conv1d

ABI=$(.venv/bin/python -c 'import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())')                                                   
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_CXX11_ABI="$ABI"

MAX_JOBS=4 \
	.venv/bin/python -m pip install --force-reinstall --no-cache-dir \
	--no-build-isolation --no-deps --no-binary=:all: \
	causal-conv1d==1.7.0


uv pip install "flash-linear-attention[cuda]"

uv pip install langdetect tblib decord

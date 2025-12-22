#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Capture the directory from where this script was called
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check if llamafactory is installed
if ! python -c "import llamafactory" 2>/dev/null; then
    echo "llamafactory not found. Installing llamafactory==0.9.3..."
    pip install llamafactory==0.9.3
else
    echo "llamafactory is already installed."
fi
# check if wrapt is installed
if ! python -c "import wrapt" 2>/dev/null; then
    echo "wrapt not found. Installing wrapt..."
    pip install wrapt
fi

# Function to check if YAML config has distill: true
check_distill_enabled() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        echo "Error: Config file $config_file not found"
        return 1
    fi

    # Check if file is YAML
    if [[ "$config_file" != *.yaml && "$config_file" != *.yml ]]; then
        echo "Error: Config file must be YAML format (.yaml or .yml)"
        return 1
    fi

    # Use grep to find distill: true in the YAML file, ignoring commented lines
    # Look for uncommented distill: true lines
    if grep -v "^[[:space:]]*#" "$config_file" | grep -qE "^[[:space:]]*distill[[:space:]]*:[[:space:]]*true[[:space:]]*$"; then
        return 0
    else
        return 1
    fi
}

# Function to check if YAML config has compress: true
check_compress_enabled() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        echo "Error: Config file $config_file not found"
        return 1
    fi

    # Check if file is YAML
    if [[ "$config_file" != *.yaml && "$config_file" != *.yml ]]; then
        echo "Error: Config file must be YAML format (.yaml or .yml)"
        return 1
    fi

    # Use grep to find compress: true in the YAML file, ignoring commented lines
    # Look for uncommented compress: true lines
    if grep -v "^[[:space:]]*#" "$config_file" | grep -qE "^[[:space:]]*compress[[:space:]]*:[[:space:]]*true[[:space:]]*$"; then
        return 0
    else
        return 1
    fi
}


# Function to extract a top-level YAML scalar value for a given key
# - Ignores commented lines
# - Trims whitespace and surrounding quotes
get_yaml_value() {
    local key="$1"
    local config_file="$2"
    awk -v k="$key" '
        BEGIN { FS = ":[[:space:]]*" }
        $0 !~ /^[[:space:]]*#/ {
            if ($1 ~ "^[[:space:]]*" k "[[:space:]]*$") {
                val = $2
                sub(/[[:space:]]*#.*/, "", val)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
                gsub(/^"|"$/, "", val)
                print val
                exit
            }
        }
    ' "$config_file"
}


# Strip surrounding matching quotes (single or double) from a string
strip_surrounding_quotes() {
    local s="$1"
    if [[ ${#s} -ge 2 ]]; then
        local first_char="${s:0:1}"
        local last_char="${s: -1}"
        if [[ "$first_char" == "$last_char" && ( "$first_char" == '"' || "$first_char" == "'" ) ]]; then
            s="${s:1:-1}"
        fi
    fi
        printf '%s\n' "$s"
}

# Function to validate valid YAML config file
validate_yaml_config() {
    local config_file="$1"

    # Check if config file exists
    if [[ ! -f "$config_file" ]]; then
        echo "Error: Config file $config_file not found"
        exit 1
    fi

    # Check if config file is a YAML file
    if [[ ! "$config_file" =~ \.(yaml|yml)$ ]]; then
        echo "Error: Config file $config_file is not a YAML file (.yaml or .yml extension required)"
        exit 1
    fi

    echo "✓ Config file validation passed: $config_file"
}


if [[ $1 == "help" ]] || [[ $1 == "-h" ]]; then
    echo "Usage: $0 <config_file.yaml> [--accelerate_config <accelerate_config.yaml>] [--use_fsdp2 true|false]"
    echo "Arguments:"
    echo "  <config_file.yaml>    YAML config file for llama_factory"
    echo "  --accelerate_config   Accelerate config file (optional)"
    echo "  --use_fsdp2           Use FSDP2 instead of FSDP1 (default: false)"
    echo "  $0 llama_config.yaml --accelerate_config ../accelerate_config/fsdp2.yaml"
    echo ""
    echo "or"
    echo ""
    echo "$0 train <config_file.yaml> "
    echo ""
    echo "Arguments:"
    echo "  train"
    echo "  <config_file.yaml>    YAML config file for llama_factory"
    echo "  --help, -h            Show help message"
    echo ""
    echo "Example:"
    echo "  $0 train llama_config.yaml"
    exit 0
fi

if [[ $1 == "train" ]] || [[ $1 == "version" ]] || [[ $1 == "help" ]]; then
    # llamafactory cli mode
    CONFIG_FILE=$2
    if [[ $1 == "version" ]] || [[ $1 == "help" ]]; then
        python $SCRIPT_DIR/llamafactory_cli.py $1
        exit 0
    fi
    # check if user is asking for -h for different option
    if [[ "$CONFIG_FILE" = "-h" ]]; then
        python $SCRIPT_DIR/llamafactory_cli.py $1 $2
        exit 0
    fi

    # Validate the config file
    validate_yaml_config "$CONFIG_FILE"

    # Extract required value from YAML config
    MODEL=$(strip_surrounding_quotes "$(get_yaml_value "model_name_or_path" "$CONFIG_FILE")")
    OUTPUT_DIR=$(strip_surrounding_quotes "$(get_yaml_value "output_dir" "$CONFIG_FILE")")
    python $SCRIPT_DIR/llamafactory_cli.py $1 $CONFIG_FILE
else
    # Run llamafactory using accelerate
    # Parse command line arguments
    CONFIG_FILE=$1
    # Move to next argument
    shift
    ACCELERATE_CONFIG=""
    USE_FSDP2="false"

    while [ $# -gt 0 ]; do
        case "$1" in
            --accelerate_config*)
            if [[ "$1" != *=* ]]; then shift; fi
            ACCELERATE_CONFIG="${1#*=}"
            ;;
            --use_fsdp2*)
            if [[ "$1" != *=* ]]; then shift; fi
            USE_FSDP2="${1#*=}"
            ;;
            *)
            echo "Unknown argument: $1"
            echo "Use bash launch_llamafactory.sh -h for usage information"
            exit 1
            ;;
        esac
        shift
        done

    if [[ -z "$FSDP_ARGS" ]]; then
        FSDP_ARGS=""
    fi

    # Validate the config file
    validate_yaml_config "$CONFIG_FILE"

    echo "=== LLaMA Factory Launcher ==="
    echo "Config file: $CONFIG_FILE"

    # Extract required value from YAML config
    MODEL=$(strip_surrounding_quotes "$(get_yaml_value "model_name_or_path" "$CONFIG_FILE")")
    OUTPUT_DIR=$(strip_surrounding_quotes "$(get_yaml_value "output_dir" "$CONFIG_FILE")")

    if [[ -z "$MODEL" || -z "$OUTPUT_DIR" ]]; then
        echo "Error: Failed to extract 'model_name_or_path' or 'output_dir' from $CONFIG_FILE"
        exit 1
    fi

    echo "Model path: $MODEL"
    echo "Output dir: $OUTPUT_DIR"

    # Check for distill: true in config
    if check_distill_enabled "$CONFIG_FILE"; then
        echo "Quantization aware distillation enabled"
        HAS_TEACHER_MODEL=true
    else
        HAS_TEACHER_MODEL=false
    fi

    # Set default accelerate config if not provided
    if [[ -z "$ACCELERATE_CONFIG" ]]; then
        if check_compress_enabled "$CONFIG_FILE"; then
            ACCELERATE_CONFIG="$SCRIPT_DIR/../accelerate_config/ddp.yaml"
        elif [[ "${USE_FSDP2,,}" == "true" ]]; then
            ACCELERATE_CONFIG="$SCRIPT_DIR/../accelerate_config/fsdp2.yaml"
        else
            ACCELERATE_CONFIG="$SCRIPT_DIR/../accelerate_config/fsdp1.yaml"
        fi
    fi

    # Add teacher model specific FSDP args if needed
    if [[ "${HAS_TEACHER_MODEL,,}" == "true" ]]; then
        FSDP_ARGS="$FSDP_ARGS --fsdp_cpu_ram_efficient_loading False"
    fi

    # Build the command
    echo "Using accelerate config: $ACCELERATE_CONFIG"
    echo "Modified FSDP args: $FSDP_ARGS"
    accelerate launch --config_file $ACCELERATE_CONFIG $FSDP_ARGS $SCRIPT_DIR/llama_factory.py $CONFIG_FILE
fi

# nanoGPT README
For more detailed installation parameters, please refer to [nanoGPT install guide](https://github.com/axonn-ai/nanoGPT).


## Perlmutter Setup

### Setup steps

1. Clone the Repository
    ```sh
    git clone https://github.com/axonn-ai/nanoGPT.git
    cd nanoGPT
    ```

2.  Create Python Environment
    ```sh
    ./scripts/create_python_env_perlmutter.sh
    ```
    > Note: You may need to modify the path and torch version in `create_python_env_perlmutter.sh`. In this paper, the torch version is 2.5.0.

3. Activate the Environment
    ```sh
    source path_to_nanogptENV/bin/activate
    ```

4. Download Data
    ```sh
    python nanoGPT/data/openwebtext/prepare.py
    ```

## Frontier Setup

### Setup steps

1. Clone the Repository
    ```sh
    git clone https://github.com/axonn-ai/nanoGPT.git
    cd nanoGPT
    ```

2.  Create Python Environment
    ```sh
    ./scripts/create_python_env_frontier.sh
    ```
    > Note: You may need to modify the WKSPC path and torch version in `create_python_env_frontier.sh`.

4. Activate the Environment
    ```sh
    source path_to_nanogptENV/bin/activate
    ```

5. Download Data
    ```sh
    python data/openwebtext/prepare.py
    ```
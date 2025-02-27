# Python Libraries

This repository contains a list of essential Python libraries for data analysts. These libraries help in data manipulation, visualization, and machine learning tasks.

## Libraries Included

1. **Pandas**: Data manipulation and analysis.
2. **NumPy**: Numerical operations on large, multi-dimensional arrays and matrices.
3. **Matplotlib**: Plotting and visualization.
4. **Seaborn**: Statistical data visualization built on top of Matplotlib.
5. **Scikit-learn**: Machine learning tools.
6. **SciPy**: Scientific and technical computing.
7. **Jupyter**: Interactive computing and notebooks.
8. **Statsmodels**: Statistical modeling and hypothesis testing.
9. **Plotly**: Interactive graphing library.
10. **BeautifulSoup**: Web scraping. etc..

## Installation

To install the required libraries, you can use the `py_libraries.txt` file included in this repository. Follow the steps below to set up your environment:


### Step : Install the Required Libraries

Once the virtual environment is activated, you can install the required libraries using the `py_libraries.txt` file. Run the following command:

```bash
pip install -r py_libraries.txt -y
```

This command will install all the libraries listed in the `py_libraries.txt` file.

## Usage

After installing the required libraries, you can start using them in your Python scripts or Jupyter notebooks. Below is an example of how to use some of these libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
etc..


```
# Crate a Python Virtual Enviournment

```sh
cd path/to/your/project
```
```sh
python -m venv source_venv
```
```sh
source_venv\Scripts\activate
```
```sh
pip install -r py_libraries -y
```
### Deactivate the Virtual Environment:

```sh
deactivate
```
### To revert the ExecutionPolicy
```sh
Set-ExecutionPolicy -ExecutionPolicy Restricted -Scope CurrentUser
```

## Contributing

If you have any suggestions or improvements, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
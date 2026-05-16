# Setup Virtual Environment
- run command to create virtual environment in python for this repo.
```bash 
	python -m venv venv
```


# Activate Virtual Environment
- Activate virutal environment
```bash
	# in Mac/Linux
	source venv/bin/activate
	# Git Bash Windows
	source venv/Scripts/activate
	# Windows
	venv/Scripts/activate
```


# Install Packages ( In Virtual Environment)
- Activate virtual evnironment
- ```bash
	pip install -r requirements.txt
```


# Connect Jupyter Notebooks to Virtual Environment
- Activate virtual environment
- Install ipykernel and setup connector.
```bash
	pip install ipykernel
	python -m ipykernel install --user --name=ai-learning
```


# Start Jupyter Notebooks
- run command to startup juypter lab
```bash
jupyter lab
```

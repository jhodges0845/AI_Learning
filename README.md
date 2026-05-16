# Setup Virtual Environment
- ```bash 
	python -m venv venv
```

# Activate Virtual Environment
- ```bash
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
- ```bash
	pip install ipykernel
	python -m ipykernel install --user --name=ai-learning
```

# Start Jupyter Notebooks
- ```bash
jupyter lab
```

#Setup Virtual Environment
- ```python 
	python -m venv venv
```

#Activate Virtual Environment
- ```python
	# in Mac/Linux
	source venv/bin/activate
	# Git Bash Windows
	source venv/Scripts/activate
	# Windows
	venv/Scripts/activate
```

#Install Packages ( In Virtual Environment)
- Activate virtual evnironment
- ```python
	pip install -r requirements.txt
```

# Connect Jupyter Notebooks to Virtual Environment
- Activate virtual environment
- ```python
	pip install ipykernel
	python -m ipykernel install --user --name=ai-learning
```

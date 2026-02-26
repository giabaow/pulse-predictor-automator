install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install black
	python -m pip install "huggingface_hub[cli]"

format:
	python -m black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

push-hub:
	# Ensure repo exists
	hf repos create giabaow/PulsePredictor-Automator --type=space || echo "Repo exists"

	# Upload App folder
	hf upload giabaow/PulsePredictor-Automator ./App

	# Upload Model folder to /Model in repo
	hf upload giabaow/PulsePredictor-Automator ./Model Model

	# Upload Results folder to /Metrics in repo
	hf upload giabaow/PulsePredictor-Automator ./Results Metrics

deploy: install push-hub
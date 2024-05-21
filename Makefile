install:
	pip	install --upgrade pip &&\
    pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./results/metrics.txt >> report.md
	echo "" >> report.md
	echo "## Confusion Matrix Plot" >> report.md
#	echo "![Confusion Matrix](./results/confusionmatrix.png)" >> report.md

#	export IMAGE_URL=$$(cml publish ./results/confusionmatrix.png)
#	echo "Published Image URL: $$IMAGE_URL"


	# Debbuging
#	if [ -f "./results/confusionmatrix.png" ]; then \
#	    echo "![](./results/confusionmatrix.png)" >> report.md; \
#	    echo "![Confusion Matrix]($$IMAGE_URL)" >> report.md; \
#	else \
#	    echo "Confusion matrix image not found." >> report.md; \
#	fi

#	ls -lh ./results/confusionmatrix.png || echo "Confusion matrix image not found."
#	cat report.md
#	ls -lh report.md

	cml comment create report.md --publish
update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
	pip install -U "huggingface_hub[cli]"
	git pull origin update
	git switch update
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload SergeyBetrays/Loan-Approval ./app --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload SergeyBetrays/Loan-Approval ./model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload SergeyBetrays/Loan-Approval ./results --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub